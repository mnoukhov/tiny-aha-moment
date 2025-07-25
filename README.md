# Playing with Reasoning, GRPO, and Countdown

## Setup

If your machine doesn't have deepspeed, wandb, or flash-attn install as below
```
pip install deepspeed==0.16.9 wandb
pip install flash-attn --no-build-isolation
```

We do logging to wandb. I highly recommend having an account for experiment tracking. Sign up on wandb.ai then in the terminal

```
wandb login
```

and follow the instructions

## Lets Reproduce Deepseek R1 (kinda) 

Jiayi Pan made a little repro of Deepseek R1's "aha moment" see the tweets \url{https://x.com/jiayi_pirate/status/1882839370505621655}

For speed, we're using a smaller model that has better mid-training (Qwen3-0.6B-base vs Qwen2.5 used by Jiayi).

The major thing is we're looking for our model to learn the format of our `<answer> </answer>` format and learn to use reasoning, backtracking, verification and all those great skills!

Look at the scores, the format correctness, how the response length changes.

Look at the specific questions and responses in the `episodes` in wandb.

Look at the `train/loss`. What's going on?

## Let's look at some issues with GRPO?

Let's look at some issues with GRPO as pointed out by this paper: \url{https://arxiv.org/abs/2503.20783}

We've got three things to check out
1. Is our prompt template actualling making our model worse?
2. Does the length normalization make our wrong answers much longer than our right answer? What happens if we remove it?
3. What effect does dividing by the standard deviation of our group do?

For the second two, look for the comment `# GRPO` to find the corresponding code

## Further exploration

What happens if we use a weaker model for our experiments? `Qwen/Qwen2.5-0.5B-base` is the old version of Qwen. Can it learn to play Countdown?

Can we figure out how good our model *will* be after training before training? One thing to look at is whether it *can* answer questions at all.
To do this, we can look at the `train/pass_at_group` metric that tracks whether there was at least 1 answer in our group of `num_responses_per_prompt` that got it right. Can we increase the number of `num_responses_per_prompt` to try and get more signal?

Let's try other models like a bigger / better version of 2.5 `Qwen/Qwen2.5-1.5B-base`. If that model doesn't work, let's take a model that has been SFTed on correct reasoning traces for countdown but with a slightly different template `CohenQu/Qwen2.5-1.5B_Countdown-v1`. What if we change the `PROMPT_TEMPLATE` to match it exactly?

## Citation

This is a modification of the nano aha moment by Amirhossein and Milad et al, check it out here \url{https://github.com/McGill-NLP/nano-aha-moment}
