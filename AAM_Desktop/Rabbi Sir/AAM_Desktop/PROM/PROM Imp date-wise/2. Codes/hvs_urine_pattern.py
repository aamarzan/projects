import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

# Load your data from Excel
excel_file = "F:\\PAHMCH\\PROM\\datasets\\microbiological\\HVS_PROM.xlsx"
df = pd.read_excel(excel_file, index_col=0, sheet_name='input')  # Assuming the first column contains row labels

# Extract row and column labels and data
ylabels = df.index.tolist()
xlabels = df.columns.tolist()
data = df.values

# Create meshgrid
N, M = data.shape
x, y = np.meshgrid(np.arange(M), np.arange(N))

# Normalize the data for size
R = data / 66

# Create rectangle
#squares = [plt.Rectangle((j - r/2, i - r/2), r, r, linewidth=2, edgecolor='none', facecolor='none') for r, j, i in zip(R.flat, x.flat, y.flat)]
#col = PatchCollection(squares, array=data.flatten(), cmap="viridis")
# Create circle
circles = [plt.Circle((j, i), r, linewidth=2, edgecolor='none', facecolor='none') for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=data.flatten(), cmap="viridis")

# Calculate the figure size based on the number of rows and columns
fig_size = (M, N)
# Plotting
fig, ax = plt.subplots(figsize=fig_size)  # Set the figure size
ax.add_collection(col)

# Set labels and ticks
ax.set_xticklabels(xlabels, rotation=45, ha='right')
ax.set(xticks=np.arange(M), yticks=np.arange(N),
       xticklabels=xlabels, yticklabels=ylabels)
ax.set_xticks(np.arange(M+1)-0.5, minor=True)
ax.set_yticks(np.arange(N+1)-0.5, minor=True)
ax.grid(which='minor')

# Add colorbar
fig.colorbar(col)

# Set aspect ratio to be equal
ax.set_aspect('equal')

plt.show()
