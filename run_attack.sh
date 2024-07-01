#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack-%j.out

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em 2_efficientnet_incrementalCIFAR100=ABCD_0 -d incrementalCIFAR100-0  
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em 2_efficientnet_incrementalCIFAR100=ABCD_1 -d incrementalCIFAR100-1 
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em 2_efficientnet_incrementalCIFAR100=ABCD_2 -d incrementalCIFAR100-2 
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em 2_efficientnet_incrementalCIFAR100=ABCD_3 -d incrementalCIFAR100-3 





