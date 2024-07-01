#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o federated-%j.out
module load container_env pytorch-gpu

# Default values
model_name="efficientnet"
base_dataset_name="CIFAR100"
num_rounds=10
num_clients=2
threads=2
weight=""
split_count=1  # Default value for the single of iterations

# Parse the command-line arguments for the main script
while getopts m:d:n:r:t:w:s: flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        d) base_dataset_name=${OPTARG};;
        n) num_clients=${OPTARG};;
        r) num_rounds=${OPTARG};;
        t) threads=${OPTARG};;
        w) weight=${OPTARG};;
        s) split_count=${OPTARG};;
    esac
done





# Loop to repeat the script execution
for ((i=0; i<split_count; i++))
do

   if [ $split_count -gt 1 ]; then
   # Construct the dataset name
        dataset_name="${base_dataset_name}_$i"
    else
        dataset_name="$base_dataset_name"
    fi

    experiment_name="${num_clients}_${model_name}_${dataset_name}"
   
    # Construct the weight flag for the -w option
    if [ $i -eq 0 ]; then
        if [ -z "$weight" ]; then
            weight_flag=""
        else
            weight_flag="-w $weight"
        fi
    else
        prev_file="${experiment_name%_*}_$((i-1)).pt"
        weight_flag="-w $prev_file"
    fi

    # Run the script for the current iteration
    echo "Running create_and_run_simulation.sh -m $model_name -d $dataset_name -n $num_clients -r $num_rounds -t $threads $weight_flag"
    ./create_and_run_simulation.sh -m $model_name -d $dataset_name -n $num_clients -r $num_rounds -t $threads $weight_flag

    
    

    if [ ! -d "./saved_models" ]; then
        # Create the directory if it does not exist
        mkdir ./saved_models
        echo "Directory 'saved_models' created."
    fi


    # Copy and rename the file
    src_path="./workspace/$experiment_name/simulate_job/app_server/FL_global_model.pt"
    dest_path="./saved_models/$experiment_name.pt"

    if cp "$src_path" "$dest_path"; then
        echo "Copied and renamed the file to $dest_path"
    else
        echo "Error: Failed to copy $src_path"
        exit 1  # Exit the script with an error code
    fi

done
