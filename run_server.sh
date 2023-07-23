#!/bin/bash

#SBATCH -c 4
#SBATCH -t 1-0 # max run time set to 1 day
#SBATCH -o /outputs/server-%j.out

module load container_env python3

port=3"${SLURM_JOB_ID:0:4}"

echo "Server is launching, the server will be available at: $(hostname):${port}"
echo "$(hostname)" > .server_hostname
echo "${port}"     > .server_port

crun -p ~/envs/ppfl/ python server.py -p "$port" -r 100 -N 3 -m efficientnet -w    # add rest of your arguments
