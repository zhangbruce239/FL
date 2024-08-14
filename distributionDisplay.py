import pandas as pd
import matplotlib.pyplot as plt
import os

# Data for each client file with total rows
data_with_totals = {
    'client1': {'total_rows': 330, 'label_distribution': [10.30, 2.73, 14.55, 6.36, 37.58, 14.85, 7.58, 6.06]},
    'client2': {'total_rows': 786, 'label_distribution': [12.21, 13.23, 17.30, 8.91, 4.96, 18.19, 17.43, 7.76]},
    'client3': {'total_rows': 614, 'label_distribution': [10.59, 3.58, 16.45, 11.40, 23.62, 6.84, 0.65, 26.87]},
    'client4': {'total_rows': 391, 'label_distribution': [9.46, 9.72, 15.09, 2.30, 19.44, 29.16, 9.97, 4.86]},
    'client5': {'total_rows': 573, 'label_distribution': [15.01, 23.56, 3.32, 14.66, 0.70, 6.46, 27.40, 8.90]},
    'client6': {'total_rows': 424, 'label_distribution': [19.34, 21.70, 8.96, 14.39, 3.07, 3.54, 9.20, 19.81]},
}

# Plot the data
labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6', 'Label 7']
colors = plt.cm.tab20.colors  # Color map for 8 different labels

fig, ax = plt.subplots(figsize=(14, 8))

bars = []
for idx, (file, data) in enumerate(data_with_totals.items()):
    total_rows = data['total_rows']
    label_distribution = data['label_distribution']
    bottom = 0
    for jdx, (label_percentage, color) in enumerate(zip(label_distribution, colors)):
        height = (label_percentage / 100) * total_rows
        bar = ax.bar(idx, height, bottom=bottom, color=color)
        bottom += height

# Setting larger fonts and font family
font = {'family': 'Times New Roman', 'size': 14}
plt.rc('font', **font)

ax.set_xticks(range(len(data_with_totals)))
# ax.set_xticklabels(data_with_totals.keys(), r, ha='right')
ax.set_xlabel('α=1', fontsize=16, fontname='Times New Roman')
ax.set_ylabel('Number of Samples', fontsize=16, fontname='Times New Roman')
# ax.set_title('Label Distribution for Each Dataset', fontsize=18, fontname='Times New Roman')
# ax.set_title('Label Distribution for Each Dataset')

# Remove the legend
# ax.legend(bars, labels, title='Labels')

# Create the results directory if it doesn't exist
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Save the plot to a file
output_path = os.path.join(output_dir, 'labelDistribution1.png')
fig.savefig(output_path)

plt.show()
