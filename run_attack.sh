#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1486dhrd@gmail.com

module load container_env  tensorflow-gpu/2.10.0

crun -p ~/envs/ppfl python attack.py -sv -c  -e 100 -e1 150 -d incremental_CIFAR100-3 -m efficientnet -mw CentralizedefficientnetincrementalCIFAR1003 -s efficientnet # add rest of your arguments
