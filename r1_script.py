import os
from pathlib import Path

import argparse
import gc
import re
import time
from typing import Any, Dict, List, Tuple, Union
import datetime

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

import wandb
from utils import (
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_free_port,
    find_last_checkpoint,
    prepare_model_inputs,
    load_model_into_vllm,
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM


SCRATCH = Path.home() / "scratch"

FORMAT_REWARD_WEIGHT = 0.1
EQUATION_REWARD_WEIGHT = 0.9


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


# Load and process dataset
def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    SYSTEM_MESSAGE: str,
    PROMPT_TEMPLATE: str,
):
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    # prefix = [
    #     {"role": "system", "content": SYSTEM_MESSAGE},
    #     {
    #         "role": "user",
    #         "content": PROMPT_TEMPLATE.format(numbers=numbers, target=target),
    #     },
    #     {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    # ]
    # input_ids = tokenizer.apply_chat_template(
    #     prefix, tokenize=True, continue_final_message=True
    # )
    # prompt = tokenizer.decode(
    #     input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )
    prompt = PROMPT_TEMPLATE.format(numbers=numbers, target=target)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {"prompt": prompt, "input_ids": input_ids}


def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: <think>...</think><answer>...</answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Synthetically prepend <think> (if your pipeline relies on that to ease matching)
        completion = "<think>" + completion

        # Strip EOS token if present
        if completion.endswith(EOS_TOKEN):
            completion = completion[: -len(EOS_TOKEN)]

        # Check if the format is correct
        # Pattern means:
        # 1) <think>...contents not including other <think> tags...</think>
        # 2) \n
        # 3) <answer>...anything...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(2).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(
    completion: str, sample: Dict[str, Any], EOS_TOKEN: str
) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(
        completion=completion, nums=nums, target=target
    )

    reward = (
        format_reward * FORMAT_REWARD_WEIGHT + equation_reward * EQUATION_REWARD_WEIGHT
    )

    metrics = {
        "format_reward": format_reward,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def create_training_episodes(
    samples: List[Dict[str, Any]],
    all_generations: List[List[int]],
    all_finish_reasons: List[str],
    tokenizer: AutoTokenizer,
    EOS_TOKEN_ID: int,
    EOS_TOKEN: str,
    GENERATIONS_PER_SAMPLE: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.

    This function processes generated responses and calculates rewards for training episodes by:
    1. Grouping generations by sample (GENERATIONS_PER_SAMPLE responses per input)
    2. Computing rewards and advantages for each response
    3. Processing response tokens (adding EOS tokens where needed)

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics

    Example:
        >>> samples = [{"input_ids": [1,2,3], "nums": [1,2,3], "target": 6}]
        >>> generations = [[4,5], [6,7], [8,9]]  # 3 generations per sample
        >>> finish_reasons = ["stop", "length", "stop"]
        >>> episodes, stats = create_training_episodes(samples, generations, finish_reasons)
        >>> episodes
        {
            'all_query_token_ids': [[1,2,3], [1,2,3], [1,2,3]],
            'all_response_token_ids': [[4,5,EOS], [6,7], [8,9,EOS]],
            'all_advantages': [[0.5,0.5,0.5], [-1.0,-1.0], [0.5,0.5,0.5]]
        }
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE))
        for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "rewards": [],
        "non_stop_rate": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(
            response_token_ids, skip_special_tokens=False
        )
        rewards_and_metrics = [
            compute_reward(resp, sample, EOS_TOKEN) for resp in responses
        ]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [
            [adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)
        ]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    reference_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: int,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.

    This function:
    1. Computes log probabilities for both policy and reference models
    2. Calculates KL divergence penalty between the models
    3. Computes policy gradient loss using advantages
    4. Combines the losses with KL coefficient

    Args:
        policy_model: The model being trained
        reference_model: The reference model for KL penalty calculation
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components:
                - policy_loss: Pure policy gradient loss
                - kl_penalty: KL divergence penalty
                - entropy: Policy entropy
    """
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    labels_mask = (labels[..., 1:] != -100).float()  # [batch_size, seq_len-1]

    with torch.no_grad():
        ref_logps = compute_token_log_probs(
            reference_model, model_inputs, TEMPERATURE
        )  # [batch_size, seq_len-1]

    logps = compute_token_log_probs(
        policy_model, model_inputs, TEMPERATURE
    )  # [batch_size, seq_len-1]

    kl_penalty = (
        torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    )  # [batch_size, seq_len-1]
    kl_penalty = kl_penalty * labels_mask  # [batch_size, seq_len-1]

    entropy = -logps.sum() / labels_mask.sum()  # scalar

    policy_loss = -logps * advantages[..., 1:]  # [batch_size, seq_len-1]
    policy_loss = policy_loss * labels_mask  # [batch_size, seq_len-1]

    loss = (
        policy_loss + KL_COEFFICIENT * kl_penalty
    ).sum() / total_response_len  # scalar

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len,
        "kl_penalty": kl_penalty.sum().item() / total_response_len,
        "entropy": entropy.item() / total_response_len,
    }

    return loss, metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train R1 model with PPO")
    parser.add_argument(
        "--kl_coeff", type=float, default=0.001, help="KL coefficient for PPO"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Model name/path"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="Learning rate for training"
    )
    parser.add_argument(
        "--liger_kernel", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.2)
    args = parser.parse_args()

    # Needed to stop DeepSpeed from complaining
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    ############################################
    # Hyperparameters
    ############################################

    # Model configuration
    MODEL_NAME = args.model_name

    # RL parameters
    # Total number of training iterations
    NUM_ITERATIONS = 500
    # Number of episodes to collect per iteration for training
    EPISODES_PER_ITERATION = 128
    # Number of responses to generate for each input prompt
    GENERATIONS_PER_SAMPLE = 4
    # Controls how much the policy can deviate from the reference model
    KL_COEFFICIENT = args.kl_coeff

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_BATCH_SIZE = 4
    # Learning rate for model updates
    LEARNING_RATE = 1e-6

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = 1024
    # Controls randomness in generation (higher = more random)
    TEMPERATURE = args.temperature
    # Nucleus sampling parameter (1.0 = disabled)
    TOP_P = 1.0
    # Top-k sampling parameter (-1 = disabled)
    TOP_K = -1  # no top k

    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    }

    model_name_short = MODEL_NAME.split("/")[-1]

    current_time = datetime.datetime.now().strftime("%d-%m_%H-%M")
    RUN_NAME = f"{model_name_short}_{current_time}"
    EXP_DIR = SCRATCH / "tiny-aha-moment" / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    # SYSTEM_MESSAGE = "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    # PROMPT_TEMPLATE = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
    SYSTEM_MESSAGE = None
    PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
            "SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
    )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    MODEL_CLS = (
        AutoLigerKernelForCausalLM if args.liger_kernel else AutoModelForCausalLM
    )

    policy_model = MODEL_CLS.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    reference_model = MODEL_CLS.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=0,
    )
    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    disable_dropout_in_model(policy_model)
    disable_dropout_in_model(reference_model)

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    reference_model.module.cpu()

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_prefix_caching=True,
        swap_space=1,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=2048,
        enable_sleep_mode=True,
    )

    # Wandb for logging
    wandb.init(
        name=RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "num_iterations": NUM_ITERATIONS,
            "episodes_per_iteration": EPISODES_PER_ITERATION,
            "rollouts_per_episode": GENERATIONS_PER_SAMPLE,
            "kl_coefficient": KL_COEFFICIENT,
            "temperature": TEMPERATURE,
        },
    )

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter
        load_model_into_vllm(policy_model, inference_engine)

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        print(f"Iteration {iteration}/{NUM_ITERATIONS}")

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################

        eval_stats = None
        if iteration % 25 == 0 and iteration > 0:
            print("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eos_token=EOS_TOKEN,
                eval_sampling_params=SamplingParams(
                    temperature=0.3,
                    max_tokens=1024,
                    n=1,
                    detokenize=True,
                    stop=["User:", EOS_TOKEN, "</answer>", "Assistant:"],
                    include_stop_str_in_output=True,
                ),
                reward_func=lambda completion, sample: compute_reward(
                    completion, sample, EOS_TOKEN
                ),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})

        #########################################################
        # Generate Episodes
        #########################################################

        # Sample training batch
        num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
        indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=GENERATIONS_PER_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_RESPONSE_TOKENS,
                detokenize=True,
                stop=["User:", EOS_TOKEN, "</answer>", "Assistant:"],
                include_stop_str_in_output=True,
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
        inference_engine.sleep(1)

        print(f"Generated {len(all_generations)} responses")
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        print(
            f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds"
        )

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples,
            all_generations,
            all_finish_reasons,
            tokenizer,
            EOS_TOKEN_ID,
            EOS_TOKEN,
            GENERATIONS_PER_SAMPLE,
        )
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device="cuda",
        )

        # Calculate losses and update model
        policy_model.train()
        reference_model.module.cuda()
        reference_model.eval()

        total_response_len = (model_inputs["labels"] != -100).sum().item()

        train_time = time.time()

        for i in trange(
            0,
            EPISODES_PER_ITERATION,
            PER_DEVICE_BATCH_SIZE,
            desc="Gradient Accumulation",
        ):
            batch = {
                k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()
            }

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                total_response_len=total_response_len,
                TEMPERATURE=TEMPERATURE,
                KL_COEFFICIENT=KL_COEFFICIENT,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else v
                )

            # Backpropagation and optimization step
            policy_model.backward(loss, scale_wrt_gas=False)

            # Free memory
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()

            policy_model.step()

        print(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
        train_metrics["learning_rate"] = policy_model.get_lr()[0]
        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
        wandb.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_reward",
            "train/reward_metrics/equation_reward",
            "eval/rewards",
            "eval/reward_metrics/format_reward",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
        print(f"KEY METRICS: {selected_metrics}")

        if iteration % 50 == 0 and iteration != 0:
            policy_model.module.save_pretrained(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model")
            )
            policy_model.save_checkpoint(
                str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed")
            )


if __name__ == "__main__":
    main()
