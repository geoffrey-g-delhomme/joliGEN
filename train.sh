#!/bin/bash

#SBATCH --job-name=joligen
#SBATCH --partition=v100
#SBATCH --gpus-per-node=8
#SBTACH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j-srun-%n.out
#SBATCH --error=./slurm/%j-srun-%n.err

python train.py --config_json $@