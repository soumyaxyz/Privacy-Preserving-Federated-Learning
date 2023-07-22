#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o centralized-%j.out

module load container_env python3

crun -p ~/envs/ppfl python attack.py -sv -n 10   -e 100 -e1 150  -w  -m efficientnet -mw EfficientNet -s efficientnet # add rest of your arguments
