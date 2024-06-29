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
root="saved_models/"
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

   
   
   # Construct the weight flag for the -w option
   if [ $i -eq 0 ]; then
       if [ -z "$weight" ]; then
            weight_flag=""
       else
            weight_flag="-w $weight"
       fi
   else
       prev_file="local_model_$((i-1))"
       weight_flag="-w $prev_file"
   fi

   # Run the simulation script
   echo "Running create_and_run_simulation.sh -m $model_name -d $dataset_name -n $num_clients -r $num_rounds -t $threads $weight_flag"
   /home/sbane002/Flare/Privacy-Preserving-Federated-Learning/create_and_run_simulation.sh -m $model_name -d $dataset_name -n $num_clients -r $num_rounds -t $threads $weight_flag

   # Copy the file to the destination
   echo "Copying trained model weights to saved_models"
   cp ./workspace/simulate_jobs/models/local_models/local_model.pt ./saved_models/local_model_$i.pt

#    # Rename the output file for the next iteration
#    mv output_filename "filename_$i"
done
