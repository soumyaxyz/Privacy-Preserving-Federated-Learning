import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from collections import Counter
import numpy as np
import pandas as pd
import pdb, traceback

# Specify the file path
file_prefix = "results/strategy_cifar10/eval_results_"
fileA_prefix = "results/strategy/confidences_"
fileB_prefix = "results/strategy/incorrect_confidences_"
fileC_prefix = "results/strategy/confidences_"
def plot_details():
    # Set labels and title
    plt.xlabel('Correctly classified by (n) clients', fontsize=18)
    plt.ylabel('Percentage', fontsize=18)
    # plt.title('Bar Plot of Value Frequencies', fontsize=18)
    plt.xticks(fontsize=18)  # Adjust the fontsize as needed
    plt.yticks(fontsize=18)  # Adjust the fontsize as needed
    
    # Convert counts to percentages
    total_samples = 5000  # Assuming you have 5000 samples
    plt.gca().yaxis.set_major_formatter(PercentFormatter(total_samples, decimals=0))

    plt.legend(fontsize=18)  # loc='lower left')

    # Display the bar plot
    plt.show()


def count_occurrences_and_plot(data, c, label):
    # Count the occurrences of each value
    value_counts = Counter(data)

    # Extract values and counts for the bar plot
    values = list(value_counts.keys())
    counts = list(value_counts.values())
    print(values)
    print(counts)
    print('printing...')
    # pdb.set_trace()
    plt.bar(values , counts,  alpha=0.5, label=f'{label} ', color=c)
    return values, counts

def read_data(file_path, fileA_path, fileB_path, fileC_path):
    # Use a context manager to open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read both lines

    # with open(fileB_path, 'r') as fileB:
    #     linesB = fileB.readlines()

    # Split each line into a list of values and merge them
    # data = []
    values = []
    counts = []
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    label = ["train samples", "test samples", "undefined"]

    
    df = pd.read_csv(fileA_path, header=None)
    average_confidence = df.mean(axis=0).tolist()

    df = pd.read_csv(fileB_path, header=None)
    average_corr_confidence = df.mean(axis=0).tolist()

    df = pd.read_csv(fileC_path, header=None)
    average_incorr_confidence = df.mean(axis=0).tolist()

    print('average_confidence')

    # pdb.set_trace()

    i=0
    for line in lines:
        data_i = []        
        data_i.extend(line.strip().split(','))
        # Convert the data to floating-point numbers
        data_i = [float(value) for value in data_i]

        

        # Convert the floating-point numbers to integers for counting
        data_i = [int(value) for value in data_i]
        values_i, counts_i = count_occurrences_and_plot(data_i, colours[i],label[i])
        i+=1        
        values.append(values_i)
        counts.append(counts_i)  
    


    # pdb.set_trace()     
    return values, counts





min_val = 30
max_val = 30
processed = False
for i in range(min_val,max_val+1):
    file_path = file_prefix+ f"{i}.csv"
    fileA_path = fileA_prefix+ f"{i}.csv" 
    fileB_path = fileB_prefix+ f"{i}.csv"
    fileC_path = fileC_prefix+ f"{i}.csv"
    print(file_path)
    try:
        values, counts = read_data(file_path, fileA_path, fileB_path, fileC_path)  
        processed = True 
    except FileNotFoundError:
        # traceback.print_exc()
        continue

if processed:
    # Create a bar plot
    plot_details()
else:
    print("File(s) not found")



