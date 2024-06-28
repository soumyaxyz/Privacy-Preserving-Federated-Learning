#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o centralized-%j.out

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python code/centralized.py -e 50 -m efficientnet -d incrementalCIFAR100_5 -no_bar #-w  add rest of your arguments