#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu 
#SBATCH --gres gpu:1
#SBATCH -o outputs/client-%j.out

module load container_env  pytorch-gpu/2.0.1

server_host="$(cat .server_hostname)"  
server_port="$(cat .server_port)" 
# if you are going to run server elsewhere
# change the two line above to your server

echo "Client is launching, client is connecting to: ${server_host}:${server_port}"

crun -p ~/envs/ppfl/ python client.py -a "$server_host" -p "$server_port" -N 4 -n 3 -m efficientnet -d CIFAR100 -hl -o 5   # add rest of your arguments
