#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o tabulate-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1486dhrd@gmail.com

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python tabulate_attacks.py