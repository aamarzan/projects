import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load the shapefile
shapefile_path = r"E:\Shapefile of Bangladesh\bgd_adm_bbs_20201113_shp\bgd_adm_bbs_20201113_SHP\bgd_admbnda_adm2_bbs_20201113.shp"
gdf = gpd.read_file(shapefile_path)

# Standardize district names (fix common mismatches)
name_mapping = {
    "Chittagong": "Chattogram" ,
    "Jessore": "Jashore" ,
    "Barisal": "Barishal",
    "Bogra": "Bogura"
}
gdf["ADM2_EN"] = gdf["ADM2_EN"].replace(name_mapping)

# Districts to highlight (using standardized names)
highlight_districts = [
    "Dhaka", "Chattogram", "Sylhet", "Bogura", "Rajshahi",
    "Khulna", "Jashore", "Rangpur", "Mymensingh", "Barishal"
]

# Create publication-quality figure
plt.figure(figsize=(24, 20), dpi=300)
ax = plt.gca()

# Set light green background for the map
ax.set_facecolor('#F2FFFC')  # Very light green

# Plot all districts first (thin gray outlines)
gdf.boundary.plot(ax=ax, linewidth=0.2, color="gray", zorder=1)

# Highlight selected districts with light color
highlighted = gdf[gdf["ADM2_EN"].isin(highlight_districts)]
highlighted.plot(ax=ax, color="#a6cee3", edgecolor="blue", linewidth=0.1, zorder=2)  # Light blue

# Add labels directly on districts
for district in highlight_districts:
    if district in highlighted["ADM2_EN"].values:
        geom = highlighted[highlighted["ADM2_EN"] == district].geometry.iloc[0]
        centroid = geom.centroid
        
        # Add label at centroid
        ax.annotate(
            text=district,
            xy=(centroid.x, centroid.y),
            ha='center',
            va='center',
            fontsize=3,  # Smaller font size
            bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1),
            zorder=4
        )

# Add publication elements with adjusted title position
plt.title("Study Sites", 
          fontsize=4.5, 
          pad=-10,  # Increased from 0 to 10 (adjust this value as needed)
          fontweight='bold',
          y=1.02)  # Optional: Slightly raises the title vertically
plt.axis('off')

# Add legend
legend_elements = [Patch(facecolor='#a6cee3', edgecolor='none', label='Study Sites')]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.5, fontsize=3)

# Add data source
plt.annotate("Data Source: BBS 2020 Administrative Boundaries", 
             xy=(0.55, 0.064), xycoords='figure fraction',
             ha='right', fontsize=3, color='#555555')



"""# Add longitude and latitude ticks around the map (add this right before plt.tight_layout())
ax.set_axis_on()  # Turn the axis back on for the ticks
ax.tick_params(axis='both', which='both', 
               bottom=False, top=False, left=False, right=False,
               labelsize=1,  # Small font size for ticks
               length=0)  # Short tick marks

# Format the ticks to show degree values
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f°E'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f°N'))"""

# Keep the map borders but remove all ticks and labels
ax.set_axis_on()  # Ensure axis is on for border control
ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
    length=0  # Ensures no tick marks are visible
)

# Remove grid lines completely (optional)
#ax.grid(False)

# Keep subtle border (as you already have)
"""for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('gray')
    spine.set_linewidth(0.2)"""

# Keep the map borders but make them very subtle
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray') 
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_linewidth(0.2)
ax.spines['top'].set_linewidth(0.2)
ax.spines['right'].set_linewidth(0.2)
ax.spines['left'].set_linewidth(0.2)

# Adjust grid lines (optional)
#ax.grid(True, linestyle='--', linewidth=0.1, color='gray', alpha=0.5)

# Adjust layout and save
plt.tight_layout()
plt.savefig("bangladesh_districts_simplified.png", 
            dpi=600, bbox_inches='tight', facecolor='white')
plt.show()