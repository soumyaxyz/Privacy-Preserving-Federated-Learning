#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o centralized-%j.out

module load container_env  pytorch-gpu/2.0.1

#crun -p ~/envs/ppfl python centralized.py -e 2 -m lgb -d Microsoft_Malware #-w  add rest of your arguments
crun -p ~/envs/ppfl python continuous.py -m CNN_malware -d Microsoft_Malware_incremental #-w  add rest of your arguments
