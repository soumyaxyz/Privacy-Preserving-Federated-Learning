#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o outputs/centralized-%j.out

module load container_env python3

crun -p ~/envs/ppfl python centralized.py xxxxxx   # add rest of your arguments
