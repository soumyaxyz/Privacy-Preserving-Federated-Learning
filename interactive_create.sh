#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
salloc -p gpu --gres=gpu:1
bash

module load container_env pytorch-gpu




crun -p ~/envs/NVFlarev2.4.0rc8 nvflare config -jt ./templates


find ./templates/sag_custom -mindepth 1 -type d -exec rm -rf {} +

mkdir -p ./templates/sag_custom/app_0
cp ./templates/reference_code/config_fed_client.conf  ./templates/sag_custom/app_0/config_fed_client.conf

mkdir -p "./templates/sag_custom/app_1"
cp ./templates/reference_code/config_fed_client.conf ./templates/sag_custom/app_1/config_fed_client.conf

mkdir -p ./templates/sag_custom/app_server
cp ./templates/reference_code/config_fed_server.conf ./templates/sag_custom/app_server/config_fed_server.conf



# Base job creation command

crun -p ~/envs/NVFlarev2.4.0rc8 nvflare job create -force -j ./jobs -w sag_custom -sd ./code/ \
-f app_0/config_fed_client.conf \
executors[0].executor.args.model_name="efficientnet" \
executors[0].executor.args.dataset_name="CIFAR10_0" \
executors[0].executor.args.num_clients=2 \
executors[1].executor.args.model_name="efficientnet" \
executors[1].executor.args.dataset_name="CIFAR10_0" \
executors[1].executor.args.num_clients=2 \
-f app_1/config_fed_client.conf \
executors[0].executor.args.model_name="efficientnet" \
executors[0].executor.args.dataset_name="CIFAR10_1" \
executors[0].executor.args.num_clients=2 \
executors[1].executor.args.model_name="efficientnet" \
executors[1].executor.args.dataset_name="CIFAR10_1" \
executors[1].executor.args.num_clients=2 \
-f app_server/config_fed_server.conf  \
components[3].args.model_name="efficientnet" \
workflows[1].args.num_rounds=10


# Run the NVFlare simulator
crun -p ~/envs/NVFlarev2.4.0rc8 nvflare simulator -n 2 -t 2 ./jobs -w ./workspace