import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Set publication-style defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Data
data = {
    "Types of Poisoning": [
        "Pesticide", "Insecticide", "Paraquat", 
        "Aluminium Phosphide", "Snakebite", "Sedatives",
        "Rat Killer", "Other Drugs", "Paracetamol",
        "Harpic", "Household Product", "Fungicide",
        "Bee Sting & Other Insect Bite", "Unknown","Methanol",
        "Kerosine-Diesel", "Street or Commuter", "Corrosives",
        "Alcohol Overdose", "Copper Sulphate"
    ],
    "Count": [
        4463, 985, 977, 603, 823, 3236, 860, 1382, 208, 
        1369, 1068, 54, 177, 3407, 33, 70, 641, 259, 
        127, 93
    ]
}

# Create DataFrame and sort
df = pd.DataFrame(data).sort_values('Count', ascending=False)
total_cases = sum(df['Count'])  # Total = 20835

# Updated Gradient (Lighter version of original sequence)
colors = [
    # Dark to Light Blues (softer)
    '#1A3A6B', '#2A4B7D', '#3A5C8F', '#4A6DA1', 
    '#5A7EB3', '#6A8FC5',
    
    # Light Blue to Cyan (brighter)
    '#7FB8E0', '#94D1F5', '#A9EAFF',
    
    # Cyan to Light Red (soft transition)
    '#FF8A8A', '#FFA3A3', '#FFBCBC',
    
    # Light Red to Yellow (warmer)
    '#FFDF80', '#FFE999', '#FFF3B3',
    
    # Yellow to Dark Green (lighter greens)
    '#A3D9A3', '#7AC07A', '#51A751', '#388E38'
]

# Create colormap
cmap = LinearSegmentedColormap.from_list("lighter_gradient", colors, N=20)

# Create figure
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.2})

# Assign colors based on position
n_bars = len(df)
positions = np.linspace(0, 1, n_bars)
bar_colors = [cmap(pos) for pos in positions]

# Plot
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.2})
bars = sns.barplot(
    data=df,
    x='Count',
    y='Types of Poisoning',
    palette=bar_colors,
    edgecolor='black',
    linewidth=0.3,
    saturation=1.0
)

# Customize
plt.title('Distribution of Poisoning Cases\n', fontsize=16, pad=5, fontweight='bold')
plt.xlabel('\nNumber of Cases', fontsize=12, labelpad=10)
plt.ylabel('')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Set x-axis limit to 110% of max count to ensure label visibility
max_count = df['Count'].max()
plt.xlim(0, max_count * 1.10)

# Value labels with percentages (adjusted for visibility)
for p in bars.patches:
    width = p.get_width()
    percentage = (width / total_cases) * 100
    bars.text(width + 0.02 * max_count,  # Dynamic padding based on max count
             p.get_y() + p.get_height()/2, 
             f'{int(width):,} ({percentage:.1f}%)',
             ha='left', 
             va='center',
             fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

# Add colorbar legend (moved further right)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=bars, orientation='vertical', pad=0.02, aspect=40)  # Reduced pad
cbar.set_label('Gradient Scale (Low → High)', rotation=270, labelpad=15)

# Footnote
#plt.annotate('*Other Poisoning includes: Chemical, Inhalation, Methanol, and miscellaneous cases',
#             xy=(0, -0.18),
#             xycoords='axes fraction',
#             ha='left',
#             va='center',
#             fontsize=9,
#             color='#555555')

plt.tight_layout()
plt.savefig('poisoning_cases_final.png', dpi=2400, bbox_inches='tight')
plt.show()