#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
salloc -p gpu --gres=gpu:1
bash

module load container_env pytorch-gpu




crun -p ~/envs/NVFlarev2.4.0rc8 nvflare config -jt ./job_templates


find ./job_templates/sag_custom -mindepth 1 -type d -exec rm -rf {} +

mkdir -p ./job_templates/sag_custom/app_0
cp ./job_templates/reference_code/config_fed_client.conf  ./job_templates/sag_custom/app_0/config_fed_client.conf

mkdir -p "./job_templates/sag_custom/app_1"
cp ./job_templates/reference_code/config_fed_client.conf ./job_templates/sag_custom/app_1/config_fed_client.conf

mkdir -p ./job_templates/sag_custom/app_server
cp ./job_templates/reference_code/config_fed_server.conf ./job_templates/sag_custom/app_server/config_fed_server.conf



# Base job creation command

crun -p ~/envs/NVFlarev2.4.0rc8 nvflare job create -force -j ./jobs -w sag_custom -sd ./code_job/ \
-f app_0/config_fed_client.conf \
executors[0].executor.args.model_name="efficientnet" \
executors[0].executor.args.dataset_name="CIFAR10_0" \
executors[1].executor.args.model_name="efficientnet" \
executors[1].executor.args.dataset_name="CIFAR10_0" \
-f app_1/config_fed_client.conf \
executors[0].executor.args.model_name="efficientnet" \
executors[0].executor.args.dataset_name="CIFAR10_1" \
executors[1].executor.args.model_name="efficientnet" \
executors[1].executor.args.dataset_name="CIFAR10_1" \
-f app_server/config_fed_server.conf  \
components[3].args.model_name="efficientnet" \
workflows[1].args.num_rounds=10


# Run the NVFlare simulator
crun -p ~/envs/NVFlarev2.4.0rc8 nvflare simulator -n 2 -t 2 ./jobs -w ./workspace