#!/bin/bash

#SBATCH --job-name=joligen
#SBATCH --partition=quadro
#SBATCH --cpus-per-task=224
#SBATCH --gpus-per-node=6
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j-srun-%n.out
#SBATCH --error=./slurm/%j-srun-%n.err

python train.py --config_json $@