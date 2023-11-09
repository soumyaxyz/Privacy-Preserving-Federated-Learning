import matplotlib.pyplot as plt

def plot(acc, priv, color, label):
    # priv = [100-x for x in priv]
    local_labels = ["0","2","5", "10"]
    plt.plot(acc, priv, c=color, marker='o', label=label )
    # Label data points
    for i in range(len(acc)):
        plt.text(acc[i] + 0.2, priv[i] - .02, f'{local_labels[i]}', fontsize=10)
        # plt.text(acc[i] + 0.2, priv[i] - .02, f'(epochs:{local_labels[i]} privacy:{priv[i]:.2f}, accuracy:{acc[i]:.2f})', fontsize=10)





plt.figure(figsize=(8, 6))

#2 epoch
acc  = [81.08,  80.81,	76.3,	71.02]
priv = [77.75,  76.85,	73.63,	66.49]

two = [acc, priv]

#5 epoch
acc  = [80.96,  83.78,  83.09,  80.64]
priv = [79.44,  76.61,  76.82,  76.67]

five = [acc, priv]

#10 epoch
acc  = [83.03,  85.79, 85.57, 84.32]
priv = [82.39,  80.83, 80.41, 78.17]

ten = [acc, priv]

#20 epoch
# acc  = [82.45, 85.75, 87.04, 86.12]
# priv = [81, 82.38, 81.89, 82.47]

acc  = [83.3, 86.41, 86.83, 86.35]
priv = [83.53, 82.69, 83.14, 82.68]

twenty = [acc, priv]



plot(two[0], two[1], 'blue', '2 epoch')
plot(five[0], five[1], 'red' ,'5 epoch')
plot(ten[0], ten[1], 'yellow', '10 epoch')
plot(twenty[0], twenty[1], 'green', '20 epoch')





# Set labels and title for the flipped axes
plt.xlabel('Accuracy (%)')
plt.ylabel('Privacy (%)')
plt.gca().invert_yaxis()
plt.title('Privacy vs. Accuracy')

# Display a grid
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.show()