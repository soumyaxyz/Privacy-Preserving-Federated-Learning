import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Specify the file path
file_prefix = "results/eval_results_"

def read_data(file_path):
    # Use a context manager to open and read the file
    with open(file_path, 'r') as file:
        data = file.read().split(',')  # Split the data into a list of values

    # Convert the data to floating-point numbers
    data = [float(value) for value in data]

    # Convert the floating-point numbers to integers for    counting
    data = [int(value) for value in data]

    # Count the occurrences of each value
    value_counts = Counter(data)

    # Extract values and counts for the bar plot
    values = list(value_counts.keys())
    counts = list(value_counts.values())

    return values, counts

max_val = 50

for i in range(1,max_val):
    file_path = file_prefix+ f"{i}.csv"
    values, counts = read_data(file_path)   
    plt.bar(np.array(values) + i * 0.02, counts, width=0.02, align='center', alpha=0.5, label=f'round {i}')

    # Create a bar plot


# Set labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Bar Plot of Value Frequencies')
plt.legend(loc='lower left')

# Display the bar plot
plt.show()
