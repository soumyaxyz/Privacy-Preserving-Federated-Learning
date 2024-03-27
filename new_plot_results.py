import numpy as np
import matplotlib.pyplot as plt


def plot_results(accuracy, privacy, title):
    # Calculate means and standard deviations
    accuracy_mean = np.mean(accuracy, axis=0)
    accuracy_std = np.std(accuracy, axis=0)
    privacy_mean = np.mean(privacy, axis=0)
    privacy_std = np.std(privacy, axis=0)

    # X-axis values
    x = np.arange(len(accuracy_mean))

    # Plotting
    plt.errorbar(x, accuracy_mean, yerr=accuracy_std, label='Accuracy', fmt='-', alpha=0.4, color='blue')
    plt.errorbar(x, privacy_mean, yerr=privacy_std, label='Privacy', fmt='-', alpha=0.4, color='red')


    # plt.plot(x, accuracy_mean, label='Accuracy', marker='o', color='blue')
    # plt.plot(x, privacy_mean, label='Privacy', marker='o', color='orange')
    
    plt.fill_between(x, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, alpha=0.1, color='blue')
    plt.fill_between(x, privacy_mean - privacy_std, privacy_mean + privacy_std, alpha=0.1, color='red')


    # Adding labels for each data point
    for i, (x_val, y_val) in enumerate(zip(x, accuracy_mean)):
        plt.text(x_val, y_val, f'{title[i]}', ha='center', va='bottom', color='blue', fontsize=15)
    for i, (x_val, y_val) in enumerate(zip(x, privacy_mean)):
        plt.text(x_val, y_val, f'{title[i]}', ha='center', va='bottom', color='red', fontsize=15)
    

    
    return x

def show_plot(x):

    # Set y-axis range
    plt.ylim(50, 100)  # Adjust these values as needed
    
    # Adding labels and title
    plt.xlabel('Training increments')
    plt.ylabel('Metrics')
    plt.title('Accuracy and Privacy')
    plt.xticks(x, ['Initial', 'Incriment 1', 'Incriment 2', 'Incriment 3'])
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # Data
    # accuracy = np.array([[57.15, 57.64, 59.57, 60.76], [55.28, 58.97, 57.5, 61.79]])
    # privacy = np.array([[67.26, 54.9, 54.27, 54.17], [64.24, 56.54, 54.25, 53.51]])
    # x = plot_results(accuracy, privacy, ['A', 'AB', 'ABC', 'ABCD'])


    # accuracy = np.array([[55.53,	58.39,	61.49,	62.52], [52.96,	58.15,	60.31,	60.44]])
    # privacy = np.array([[57.3,	54.63,	54,	79.2], [58.86,	54.31,	54,	75.23]])
    # x = plot_results(accuracy, privacy, [ 'B', 'BC', 'BCD', 'BCDA'])


    accuracy = np.array([[53.97,	59.62,	59.97,	60.93], [53.53,	59.02,	58.86,	63.37]])
    privacy = np.array([[56,	68.3,	55.72,	56.23], [54.78,	73.69,	57.91,	56.23]])
    x = plot_results(accuracy, privacy, [ 'D', 'DA', 'DAB', 'DABC'])




    # accuracy = np.array([[57.15, 57.64, 59.57, 60.76], [55.28, 58.97, 57.5, 61.79], [55.53,	58.39,	61.49,	62.52], [52.96,	58.15,	60.31,	60.44], [53.97,	59.62,	59.97,	60.93], [53.53,	59.02,	58.86,	63.37]])
    # privacy = np.array([[67.26, 54.9, 54.27, 54.17], [64.24, 56.54, 54.25, 53.51],[57.3,	54.63,	54,	79.2], [58.86,	54.31,	54,	75.23], [56,	68.3,	55.72,	56.23]])
    # x = plot_results(accuracy, privacy, ['', '', '', ''])


   



    show_plot(x)


    
