
import argparse
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def abline(slope, intercept, color = 'red'):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color = color)

def plot_data(filename, sort=False, trendline=False):
    # Read the CSV file
    data = pd.read_csv(filename)
    data_len = len(data)
    if sort:
        data.sort_values(data.columns[1], axis=0, inplace=True) # type: ignore

    # Get the second column
    col2 = data.iloc[:, 1]
    
    if trendline:
        
        zeros = data.loc[data['label'] == 0.0]
        avg_zeros = np.mean(zeros['data'])
        plt.plot( [*range(data_len)] , [avg_zeros]*data_len, color=plt.cm.viridis(0), label='zeros') # type: ignore
        

        ones = data.loc[data['label'] == 1.0]   
        avg_ones = np.mean(ones['data'])
        plt.plot( [*range(data_len)] , [avg_ones]*data_len, color=plt.cm.viridis(256), label='ones') # type: ignore
        # pdb.set_trace()
        plt.legend()

        

    # Plot the data with colors based on the second column
    plt.scatter([*range(len(data))], data.iloc[:, 0], c=col2, cmap='viridis')

    # Add a colorbar
    plt.colorbar()

    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-f', '--filename', type=str, default=None, help='data filename to be plotted')
    parser.add_argument('-s', '--sort', action='store_true', help='sort the plot according to labels')
    parser.add_argument('-t', '--trendline',  action='store_true', help='plot trendline')
    args = parser.parse_args()

    plot_data(args.filename, args.sort, args.trendline)






if __name__ == "__main__":
    main()
