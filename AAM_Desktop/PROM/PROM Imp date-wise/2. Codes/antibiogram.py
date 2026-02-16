import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load your data from Excel
excel_file = "D:\\Users\\shati\\Desktop\\gram_anti.xlsx"
df = pd.read_excel(excel_file, sheet_name='prom_urine')  # Assuming the first column contains row labels

# Define custom color scheme
colors_low = 'red'
colors_mid = 'yellow'
colors_high = 'blue'
colors_nan = 'white'

# Custom colormap with three colors
cmap = LinearSegmentedColormap.from_list('custom', [colors_low, colors_mid, colors_high])

# Create a figure and axis
fig, ax = plt.subplots()

# Loop through each row in the data
for i, row in df.iterrows():
    # Replace NaN values with a placeholder (e.g., -1) and convert to a numpy array
    values = row[3:].replace(np.nan, -1).values

    # Assign colors based on thresholds
    conditions = [
        (values == -1),
        (values < row['Threshold_1']),
        ((values >= row['Threshold_1']) & (values <= row['Threshold_2'])),
        (values > row['Threshold_2']),
    ]
    color_values = [colors_nan, colors_low, colors_mid, colors_high]
    colors = np.select(conditions, color_values)

    # Plot the heatmap for each row
    heatmap = ax.scatter(range(3, len(row)), [i] * (len(row)-3), c=colors, cmap=cmap,
                         marker='s', s=100, vmin=-2, vmax=55,  # Adjust min and max values as needed
                         edgecolor='black', linewidths=0.5)

    # Add annotations for actual values
    #for x, val in zip(range(3, len(row)), values):
        #ax.text(x, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color='black')


# Customize the colorbar
cbar = plt.colorbar(heatmap, orientation='vertical')
cbar.set_label('Values')

# Set y-axis ticks and labels
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['Antibiotic'])

# Remove x-axis ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])

# Set x-axis label
ax.set_xlabel('Values')

# Show the plot
plt.show()
