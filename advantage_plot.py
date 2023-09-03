import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

class ItemGenerator:
    def __init__(self, itemstyles):
        self.itemstyles = itemstyles
        self.index = 0

    def next(self):
        current_index = self.index
        self.index = (self.index + 1) % len(self.itemstyles)
        return self.itemstyles[current_index]

  
@dataclass
class DataPoints:
    name: str
    advantage: list
    x_points: list

    bar_mode: bool = False

    def plotAdvantage(self, label, color, offset):
        if self.bar_mode:
            return self.plotAdvantageBar(label, color, offset)
        x_scaled = np.linspace(min(self.x_points), max(self.x_points), 100) # type: ignore
        advantage_interp = np.interp(x_scaled, self.x_points, self.advantage)

        # Generating a random linestyle
        random_linestyle = lineStyles.next()
        random_markerstyle =markerStyles.next()

        # plt.plot(x_scaled, advantage_interp, color=color, linewidth=7, linestyle=random_linestyle)
        plt.plot(self.x_points, self.advantage,  marker =random_markerstyle, markersize=15, label=label,color=color, linewidth=7, linestyle=random_linestyle)
    
    def plotAdvantageBar(self, label, color, offset):
        x_positions = np.array(self.x_points) + offset
        plt.bar(x_positions, self.advantage, width=0.2, align='center', label=label, color=color, alpha=0.6)

markerStyles = ItemGenerator(['o', '<', 's', 'D', '^', 'v', 'p', '*', 'h', '+', 'x'])
# ineStyles = ItemGenerator(['-', '--', '-.', ':'])
lineStyles = ItemGenerator(['solid', 'dotted', 'dashdot', 'dashed'])



# Data
# x_points = [1, 3, 5] # bar mode
x_points = [2, 5, 10]
CIFAR10 = DataPoints(name="CIFAR10", advantage=[0.03233, 0.0005, -0.0137], x_points=x_points)
CIFAR100 = DataPoints(name="CIFAR100", advantage=[0.02, 0.012, 0.012], x_points=x_points)
MNIST = DataPoints(name="MNIST", advantage=[0.00797, 0.00583, 0.00296], x_points=x_points)
FashionMNIST = DataPoints(name="FashionMNIST", advantage=[0.06, 0.056, 0.01099], x_points=x_points)

# Increase text size and set the figure size for A4 paper
plt.rcParams.update({'font.size': 40})  # Adjust font size
fig, ax = plt.subplots(figsize=(16, 16))  # Adjust the figure size for A4 double column
# plt.yscale('log')  # Set the y-axis scale

# Plotting
CIFAR10.plotAdvantage(label="CIFAR10", color="blue", offset=-0.45)
CIFAR100.plotAdvantage(label="CIFAR100", color="green", offset=-0.15)
MNIST.plotAdvantage(label="MNIST", color="red", offset=0.15)
FashionMNIST.plotAdvantage(label="FashionMNIST", color="orange", offset=0.45)

plt.axhline(y=0, color='gray', linestyle='--')  # Add zero advantage line
plt.xlim(x_points[0]-.06, x_points[-1]+.06)
plt.ylabel('Advantage')
plt.xlabel('FL Clients')
plt.legend(fontsize='28')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.xticks(x_points, ['2', '5', '10'])
# plt.yticks(np.arange(-0.02, 0.08, 0.01))
# Reduce tick size
plt.tick_params(axis='both', which='both', labelsize=28)  # You can adjust the labelsize as needed
plt.grid(True)
# plt.tight_layout()  # Ensures proper spacing
plt.savefig("advantage_plot.pdf")  # Save the plot as PDF for high-quality reproduction
plt.show()
