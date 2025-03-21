#!/bin/bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH -p main

module load python/3.10
module load cudatoolkit/12.4
export WANDB_PROJECT=r1
export WANDB_ENTITY=mila-language-drift
export HF_HOME=/network/scratch/n/noukhovm/huggingface

uv run r1_script.py
