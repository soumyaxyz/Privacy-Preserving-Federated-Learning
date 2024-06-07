#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o centralized-%j.out

module load container_env  tensorflow-gpu/2.10.0

crun -p ~/envs/ppfl python centralized.py -d CIFAR10 -m efficientnet -em FL_global_model # add rest of your arguments
