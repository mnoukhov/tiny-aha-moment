#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH -p main

module load python/3.10
module load cudatoolkit/12.4
export WANDB_PROJECT=r1-aha-moment
export WANDB_ENTITY=mila-language-drift
export HF_HOME=/network/scratch/n/noukhovm/huggingface

uv run r1_script.py --model_name Qwen/Qwen2.5-1.5B-Instruct --liger_kernel
