#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o attack-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=saham001@odu.edu

module load container_env  pytorch-gpu/2.0.1

crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em CentralizedefficientnetincrementalCIFAR1000 -d incrementalCIFAR100-0  
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em CentralizedefficientnetincrementalCIFAR1001 -d incrementalCIFAR100-1 
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em CentralizedefficientnetincrementalCIFAR1002 -d incrementalCIFAR100-2 
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em CentralizedefficientnetincrementalCIFAR1003 -d incrementalCIFAR100-3 
crun -p ~/envs/ppfl python code/membership_inference_attack.py  -m efficientnet -em CentralizedefficientnetincrementalCIFAR1004 -d incrementalCIFAR100-4




