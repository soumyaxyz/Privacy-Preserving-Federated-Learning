#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
module load container_env pytorch-gpu

# Default values
model_name="efficientnet"
base_dataset_name="CIFAR100"
num_rounds=10
num_clients=2
threads=2
root="saved_models/"
weight=""
split_count=5  # Default value for the number of iterations

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
   # Construct the dataset name
   dataset_name="${base_dataset_name}_$i"
   
   # Construct the weight flag for the -w option
   if [ $i -eq 0 ]; then
       current_weight=$weight
   else
       prev_file="filename_$((i-1))"
       current_weight="-w $prev_file"
   fi

   # Run the simulation script
   echo "Running create_and_run_simulation.sh -m $model_name -d $dataset_name -n $num_clients -r $num_rounds -t $threads -w $current_weight"
   create_and_run_simulation.sh -m $model_name -d $dataset_name -n $num_clients -r $num_rounds -t $threads -w $current_weight

   # Copy the file to the destination
   echo "Copying file to destination"
   cp file dest

   # Rename the output file for the next iteration
   mv output_filename "filename_$i"
done
