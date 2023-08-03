#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o centralized-%j.out

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python centralized.py -d CIFAR100 -m efficientnet -w  # add rest of your arguments
