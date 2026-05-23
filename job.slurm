#!/bin/bash
#SBATCH --job-name=cats_ai
#SBATCH --output=slurm/cats_ai%j.out
#SBATCH --error=slurm/cats_ai%j.err
#SBATCH --partition=main
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64GB

uv run python main.py
