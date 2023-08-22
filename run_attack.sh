#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1486dhrd@gmail.com

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python attack.py -sv -n 10 -c  -e 100 -e1 150  -w  -c  -m efficientnet -mw CentralizedEfficientNet -s efficientnet # add rest of your arguments
crun -p ~/envs/ppfl python attack.py -sv -n 10 -c  -e 100 -e1 150  -w  -c  -m efficientnet -mw Federated2efficientnetCIFAR10 -s efficientnet # add rest of your arguments
crun -p ~/envs/ppfl python attack.py -sv -n 10 -c  -e 100 -e1 150  -w  -c  -m efficientnet -mw Federated3efficientnetCIFAR10 -s efficientnet # add rest of your arguments
