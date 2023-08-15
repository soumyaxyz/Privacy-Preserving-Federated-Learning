#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o outputs/centralized-%j.out

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python attack.py -sv -n 10   -e 100 -e1 150  -w  -m efficientnet -mw Federated3efficientnetCIFAR10 -s efficientnet # add rest of your arguments
