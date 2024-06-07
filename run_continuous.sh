#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o continuous2-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aayushkapoor34@gmail.com

module load container_env  tensorflow-gpu/2.10.0

crun -p ~/envs/ppfl python continuous.py -m efficientnet -d incremental_SVHN-4 -em CentralizedefficientnetincrementalSVHN4
