import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pdb

# Specify the file path
file_prefix = "results/strategy_cifar10/eval_results_"




def count_occurrences(data, c):
    # Count the occurrences of each value
    value_counts = Counter(data)

    # Extract values and counts for the bar plot
    values = list(value_counts.keys())
    counts = list(value_counts.values())
    # print(values)
    # print(counts)
    print('printing...')
    # pdb.set_trace()
    plt.bar(values , counts,  alpha=0.5, label=f'round ', color=c)
    return values, counts

def read_data(file_path):
    # Use a context manager to open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read both lines

    # Split each line into a list of values and merge them
    # data = []
    values = []
    counts = []
    colours = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    i=0
    for line in lines:
        data_i = []        
        data_i.extend(line.strip().split(','))
        # Convert the data to floating-point numbers
        data_i = [float(value) for value in data_i]

        # Convert the floating-point numbers to integers for counting
        data_i = [int(value) for value in data_i]
        values_i, counts_i = count_occurrences(data_i, colours[i])
        i+=1        
        values.append(values_i)
        counts.append(counts_i)  
        # pdb.set_trace()    
        # print(values, counts)   
    return values, counts





min_val = 100
max_val = 100

for i in range(min_val,max_val+1):
    file_path = file_prefix+ f"{i}.csv"
    try:
        values, counts = read_data(file_path)   
        # plt.bar(np.array(values[0]) + i * 0.02, counts[0], width=0.02, align='center', alpha=0.5, label=f'round {i} test', color="tab:red")
        # plt.bar(np.array(values[1]) + i * 0.02, counts[1], width=0.02, align='center', alpha=0.5, label=f'round {i} train', color="tab:blue")
    except FileNotFoundError:
        continue


    # Create a bar plot


# Set labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Bar Plot of Value Frequencies')
plt.legend()#loc='lower left')

# Display the bar plot
plt.show()

