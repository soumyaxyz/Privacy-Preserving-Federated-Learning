import pandas as pd
import matplotlib.pyplot as plt
import pdb

# Specify the CSV file containing the data for the bar plots
csv_file_name = "results/strategy_cifar10/confidences_10.csv"

# Read the data from the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_name, header=None)




# Create a single figure to superimpose the bar plots
plt.figure()

# Iterate through the rows of the DataFrame and superimpose the bar plots
for i, row in df.iterrows():
    row = row[:100]
    print(len(row))
    # pdb.set_trace()
    plt.bar(range(len(row)), row, alpha=0.5, label=f'Bar Plot {i}')

plt.title('Superimposed Bar Plots')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()