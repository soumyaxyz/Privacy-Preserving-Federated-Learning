#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
module load container_env pytorch-gpu

# Default values
model_name="efficientnet"
dataset_name="CIFAR10"
num_rounds=10
num_clients=2
threads=2

# Parse flagged arguments
while getopts m:d:n:r:t: flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        d) dataset_name=${OPTARG};;
        n) num_clients=${OPTARG};;
        r) num_rounds=${OPTARG};;
        t) threads=${OPTARG};;
    esac
done
crun -p ~/envs/NVFlarev2.4.0rc8 nvflare config -jt ./templates

# Base template directory
template_dir="./templates/sag_custom"

# Delete all subdirectories in sag_custom
find "$template_dir" -mindepth 1 -type d -exec rm -rf {} +

reference_code_dir="./templates/reference_code"

# Create client app directories dynamically
for ((i=0; i<$num_clients; i++))
do
    client_dir="$template_dir/app_$i"
    mkdir -p "$client_dir"
    cp "$reference_code_dir/config_fed_client.conf" "$client_dir/config_fed_client.conf"
done

# Create meta.conf content
meta_conf_content="{
  name = \"sag_custom\"
  resource_spec = {}
  deploy_map {
    app_server = [\"server\"]"
for ((i=0; i<$num_clients; i++))
do
    meta_conf_content+="\n    app_$i = [\"site-$(($i + 1))\"]"
done
meta_conf_content+="\n  }
  min_clients = $num_clients
  mandatory_clients = []
}"

# Write meta.conf content to file
echo -e "$meta_conf_content" > "$template_dir/meta.conf"


# Ensure the server app directory exists
server_dir="$template_dir/app_server"
mkdir -p "$server_dir"
cp "$reference_code_dir/config_fed_server.conf" "$server_dir/config_fed_server.conf"

# Base job creation command
command="crun -p ~/envs/NVFlarev2.4.0rc8 nvflare job create -force -j ./jobs -w $template_dir -sd ./code/"
# Loop through each client and add their specific configurations
for ((i=0; i<$num_clients; i++))
do
    app_name="app_$i"
    command+=" -f $app_name/config_fed_client.conf executors[0].executor.args.model_name=\"$model_name\"   executors[0].executor.args.dataset_name=\"${dataset_name}_$i\"     executors[0].executor.args.num_clients=$num_clients    executors[1].executor.args.model_name=\"$model_name\"     executors[1].executor.args.dataset_name=\"${dataset_name}_$i\"    executors[0].executor.args.num_clients=$num_clients"
done

# Add server configurations
command+=" -f app_server/config_fed_server.conf components[3].args.model_name=\"$model_name\" workflows[1].args.num_rounds=$num_rounds"

# Execute the command
eval $command

echo "job created successfully!\n\n"

# Run the NVFlare simulator
crun -p ~/envs/NVFlarev2.4.0rc8 nvflare simulator -n $num_clients -t $threads ./jobs -w ./workspace