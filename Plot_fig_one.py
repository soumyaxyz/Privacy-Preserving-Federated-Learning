# import matplotlib.pyplot as plt

# def plot(acc, priv, color, label, linestyle='-'):
#     local_labels = ["2", "5", "10", "20"]
#     # acc = acc[1:]
#     # priv = priv[1:]
#     # local_labels = local_labels[1:]
#     plt.plot(acc, priv, c=color, marker='o', linestyle=linestyle, label=label)
#     # Label data points
#     for i in range(len(acc)):
#         plt.text(acc[i] + 0.2, priv[i] - .02, f'{local_labels[i]}', fontsize=10)

# # Data for 2 epoch
# acc_two = [81.08, 80.81, 76.3, 71.02]
# priv_two = [77.75, 76.85, 73.63, 66.49]

# # Data for 5 epoch
# acc_five = [80.96, 83.78, 83.09, 80.64]
# priv_five = [79.44, 76.61, 76.82, 76.67]

# # Data for 10 epoch
# acc_ten = [83.03, 85.79, 85.57, 84.32]
# priv_ten = [82.39, 80.83, 80.41, 78.17]


# # #20 epoch
# # # acc  = [82.45, 85.75, 87.04, 86.12]
# # # priv = [81, 82.38, 81.89, 82.47]

# # Data for 20 epoch
# acc_twenty = [83.3, 86.41, 86.83, 86.35]
# priv_twenty = [83.53, 82.69, 83.14, 82.68]

# plt.figure(figsize=(8, 6))

# # Plot each dataset with lines
# # plot(acc_two, priv_two, 'blue', '2 epoch' )
# # plot(acc_five, priv_five, 'red', '5 epoch' )
# # plot(acc_ten, priv_ten, 'yellow', '10 epoch' )
# # plot(acc_twenty, priv_twenty, 'green', '20 epoch')

# # Connect specific points across the plots with dotted lines

# plot([acc_two[0], acc_five[0], acc_ten[0], acc_twenty[0]],
#          [priv_two[0], priv_five[0], priv_ten[0], priv_twenty[0]], 'r', label='Centralized')

# plot([acc_two[1], acc_five[1], acc_ten[1], acc_twenty[1]],
#          [priv_two[1], priv_five[1], priv_ten[1], priv_twenty[1]], 'r', label='Connect 2 clients')

# plot([acc_two[2], acc_five[2], acc_ten[2], acc_twenty[2]],
#          [priv_two[2], priv_five[2], priv_ten[2], priv_twenty[2]], 'b', label='Connect 5 clients')

# plot([acc_two[3], acc_five[3], acc_ten[3], acc_twenty[3]],
#          [priv_two[3], priv_five[3], priv_ten[3], priv_twenty[3]], 'g', label='Connect 10 clients')

# # Set labels and title for the flipped axes
# plt.xlabel('Accuracy (%)')
# plt.ylabel('Privacy (%)')
# plt.gca().invert_yaxis()
# plt.title('Privacy vs. Accuracy')

# # Display a grid
# plt.grid(True)

# # Add a legend
# plt.legend()

# # Show the plot
# plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator



def plot(ax, acc, priv, epochs, color, label, linestyle='-'):
    ax.plot(acc, priv, epochs, c=color, marker='o', linestyle=linestyle, label=label)
    # Label data points
    # for i in range(len(acc)):
    #     ax.text(acc[i] + 0.2, priv[i] - .02, epochs[i] + 0.2, f'{epochs[i]}', fontsize=10)


# Data for centralized
acc_central = [80.81,	83.78,	85.79,	86.41]
priv_central = [76.85,	76.61,	83,	82.69]
epoch_central = [2, 5, 10, 20]

# Data for 2 client
acc_two = [76.3,	83.09,	85.57,	86.83]
priv_two = [73.63,	76.82,	80.41,	83.14]
epochs_two = [2, 5, 10, 20]

# Data for 5 client
acc_five = [71.02,	80.64,	84.32,	86.35]
priv_five = [66.49,	76.67,	78.17,	82.68]
epochs_five = [2, 5, 10, 20]

# Data for 10 client
acc_ten = [81.08,	80.96,	83.03,	83.3]
priv_ten = [77.75,	79.44,	82.39,	83.53]
epochs_ten = [2, 5, 10, 20]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Connect specific points across the plots with dotted lines
plot(ax, acc_central, priv_central, epoch_central, 'r', 'Centralized')
plot(ax, acc_two, priv_two, epochs_two, 'y', 'FL 2 clients')
plot(ax, acc_five, priv_five, epochs_five, 'b', 'FL 5 clients')
plot(ax, acc_ten, priv_ten, epochs_ten, 'g', 'FL 10 clients')

# Set labels and title for the flipped axes
ax.set_xlabel('Accuracy (%)')
ax.set_ylabel('Privacy (%)')
ax.set_zlabel('Rounds')
# ax.invert_yaxis()

# Set integer ticks for Accuracy and Privacy axes
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.title('Privacy vs. Accuracy vs. Epochs')

# Add a legend
ax.legend()

# Show the plot
plt.show()
