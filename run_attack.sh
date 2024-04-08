#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1486dhrd@gmail.com

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python attack.py -sv -n 10 -c  -e 100 -e1 150  -at -d Microsoft_Malware_incremental-1 -m CNN_malware -mw CentralizedCNNmalwareMicrosoftMalwareincremental1 -s CNN_malware 
