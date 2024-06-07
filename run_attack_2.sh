#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack2-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aayushkapoor34@gmail.com

module load container_env  tensorflow-gpu/2.10.0

crun -p ~/envs/ppfl python attack_2.py -d incremental_SVHN-4 -m efficientnet # add rest of your arguments
