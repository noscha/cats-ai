#!/bin/bash
#SBATCH --job-name=cats_ai
#SBATCH --output=slurm/cats_ai%j.out
#SBATCH --error=slurm/cats_ai%j.err
#SBATCH --partition=main
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-a100-[1-4]
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64GB

DATA_DIR=/scratch/cats-ai
rsync -av /mnt/stud/home/nschager/cats-ai/CarCrash ${DATA_DIR}/

export PYTHONUNBUFFERED=1

uv sync
srun uv run python main.py
