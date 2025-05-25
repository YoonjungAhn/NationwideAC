# !/usr/bin/env Python3
# -*- Coding: utf-8 -*-
"""
Created on Fri Apr  4 14:44:59 2025

@author: y943a214
"""

import pandas as pd
import geopandas as gpd
df = pd.read_csv("<your_path_here>")
df['GEOID'] = df['GEOID'].apply(lambda x: f"{x:011d}")

tracboundary = gpd.read_file("<your_path_here>")

ctboundary = gpd.read_file("<your_path_here>")
ctboundary.rename(columns={'GEOID': 'CBSA_GEOID', 'NAME':"MetroName"}, inplace=True)
ctboundary=ctboundary[['CBSA_GEOID',"MetroName", 'geometry']]

joined = gpd.sjoin( tracboundary,ctboundary) #,  how='inner', predicate='within')



DF = pd.merge(df, joined, on = ['GEOID'] )


selectdf = DF.loc[(DF['MetroName']== 'San Francisco-Oakland-Berkeley, CA')&(DF['ACtype']=='Central')]
gdf = gpd.GeoDataFrame(selectdf, geometry='geometry')


# Plotting
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gdf.plot(column='Percentage', cmap='Blues', legend=True, ax=ax,
         linewidth=0.01,  # Adjusts the thickness of the borders
         edgecolor='gray')
ax.set_title('Map of Percentage')
ax.set_ylabel('test', fontsize=12, rotation=0, ha='right', va='center', labelpad=40)
ax.set_axis_off()  # Removes the axis around the map
plt.show()



import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches



# Define The cities and their corresponding subplots
cities = ['Dallas-Fort Worth-Arlington, TX', 'Houston-The Woodlands-Sugar Land, TX']

cities = ['Dallas-Fort Worth-Arlington, TX', 'Houston-The Woodlands-Sugar Land, TX',
          'Chicago-Naperville-Elgin, IL-IN-WI', 'New York-Newark-Jersey City, NY-NJ-PA',
          'Los Angeles-Long Beach-Anaheim, CA', 'Atlanta-Sandy Springs-Alpharetta, GA',
          'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD', 'San Francisco-Oakland-Berkeley, CA',
          'Jacksonville, FL', 'New Orleans-Metairie, LA']


# AC Types
ac_types = ['Central', 'NoAC', 'Others', 'Evaporative Cooler']

# Prepare The figure layout
fig, axs = plt.subplots(len(cities), len(ac_types), figsize=(20, 20), constrained_layout=True)

for i, city in enumerate(cities):
    for j, ac_type in enumerate(ac_types):
        # Filter The dataframe for each city and AC type
        selectdf = DF.loc[(DF['MetroName'] == city) & (DF['ACtype'] == ac_type)]
        gdf = gpd.GeoDataFrame(selectdf, geometry='geometry')
        boundary_gdf = joined.loc[(joined['MetroName'] == city)]  # Ensure 'joined' is correctly defined above

        if gdf.empty:
            axs[i, j].set_visible(False)
        else:
            # Plot Boundaries
            boundary_gdf.plot(ax=axs[i, j], color='gray', linewidth=0.01, edgecolor='lightgray')

            # Plot The main data with percentage color
            gdf.plot(column='Percentage', cmap='coolwarm_r', legend=False,
                     vmin=0, vmax=100, linewidth=0.01, edgecolor='lightgray', ax=axs[i, j])



            # Set City titles on the y-axis label
            if j == 0:  # Only for the first column
                axs[i, j].set_ylabel(f'{city}', fontsize=20, rotation=0, ha='right', va='center', labelpad=40)

            # Set AC Type titles on the top
            if i == 0:
                axs[0, j].set_title(ac_type,fontsize=20, pad=20)  # Adds a pad to elevate the AC type title above the plot

            # Remove Axes borders
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)

            # Hide X and y axis ticks
            axs[i, j].xaxis.set_ticks([])
            axs[i, j].yaxis.set_ticks([])


# Add Custom colorbars in the last row of the grid
for j, ac_type in enumerate(ac_types):
    divider = make_axes_locatable(axs[-1, j])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    vmax = 20 if ac_type == 'Evaporative Cooler' else 100
    colorbar = plt.cm.ScalarMappable(cmap='coolwarm_r', norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(colorbar, cax=cax, orientation='horizontal', shrink = 0.3)
    cax.set_xlabel('Percentage')
    cb.outline.set_edgecolor('none')  # Remove the colorbar outline


# Create A custom legend for NA
na_patch = mpatches.Patch(color='gray', label='NA')
fig.legend(handles=[na_patch], loc='lower left',  bbox_to_anchor=(0.2, -0.01), fontsize=20)


fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')

plt.show()


# Adding a dot on downtown areas

from shapely.geometry import Point

# Define Downtown coordinates for each city (example coordinates — update with actual downtowns)
downtown_coords = {
    'Dallas-Fort Worth-Arlington, TX': Point(-96.7970, 32.7767),
    'Houston-The Woodlands-Sugar Land, TX': Point(-95.3698, 29.7604),
    'Chicago-Naperville-Elgin, IL-IN-WI': Point(-87.6298, 41.8781),
    'New York-Newark-Jersey City, NY-NJ-PA': Point(-74.0060, 40.7128),
    'Los Angeles-Long Beach-Anaheim, CA': Point(-118.2437, 34.0522),
    'Atlanta-Sandy Springs-Alpharetta, GA': Point(-84.3880, 33.7490),
    'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD': Point(-75.1652, 39.9526),
    'San Francisco-Oakland-Berkeley, CA': Point(-122.4194, 37.7749),
    'Jacksonville, FL': Point(-81.6557, 30.3322),
    'New Orleans-Metairie, LA': Point(-90.0715, 29.9511),
}


# Create Downtown GeoDataFrame and convert CRS later
downtown_gdf = gpd.GeoDataFrame({'MetroName': list(downtown_coords.keys())},
                                geometry=list(downtown_coords.values()), crs="EPSG:4326")

# Prepare The figure
fig, axs = plt.subplots(len(cities), len(ac_types), figsize=(20, 20), constrained_layout=True)

# Loop Through each city and AC type
for i, city in enumerate(cities):
    for j, ac_type in enumerate(ac_types):
        selectdf = DF.loc[(DF['MetroName'] == city) & (DF['ACtype'] == ac_type)]
        gdf = gpd.GeoDataFrame(selectdf, geometry='geometry')
        boundary_gdf = joined.loc[(joined['MetroName'] == city)]

        if gdf.empty:
            axs[i, j].set_visible(False)
        else:
            # Convert Downtown point to match CRS
            city_point = downtown_gdf[downtown_gdf['MetroName'] == city]
            city_point = city_point.to_crs(gdf.crs)

            # Plot Boundaries
            boundary_gdf.plot(ax=axs[i, j], color='gray', linewidth=0.01, edgecolor='lightgray')

            # Plot Data by percentage
            gdf.plot(column='Percentage', cmap='coolwarm_r', legend=False,
                     vmin=0, vmax=100, linewidth=0.01, edgecolor='lightgray', ax=axs[i, j])

            # Plot Downtown red dot
            city_point.plot(ax=axs[i, j], color='black', markersize=20, zorder=10)

            # Y-axis City labels
            if j == 0:
                axs[i, j].set_ylabel(f'{city}', fontsize=20, rotation=0, ha='right', va='center', labelpad=40)

            # Top AC Type titles
            if i == 0:
                axs[0, j].set_title(ac_type, fontsize=20, pad=20)

            # Hide Axes ticks and borders
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            axs[i, j].xaxis.set_ticks([])
            axs[i, j].yaxis.set_ticks([])

# Add Colorbars
for j, ac_type in enumerate(ac_types):
    divider = make_axes_locatable(axs[-1, j])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    vmax = 20 if ac_type == 'Evaporative Cooler' else 100
    colorbar = plt.cm.ScalarMappable(cmap='coolwarm_r', norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(colorbar, cax=cax, orientation='horizontal', shrink=0.3)
    cax.set_xlabel('Percentage')
    cb.outline.set_edgecolor('none')

# Add Custom legend for NA
na_patch = mpatches.Patch(color='gray', label='NA')
fig.legend(handles=[na_patch], loc='lower left', bbox_to_anchor=(0.2, -0.01), fontsize=20)

fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')

plt.show()




# United States map (county level)
df = pd.read_csv("<your_path_here>")
df['GEOID'] = df['GEOID'].apply(lambda x: f"{x:011d}")
df['GEOIDCN'] = df['GEOID'].str[0:5]

# Group By 'GEOIDCN' and 'ACtype', and sum 'Count'
grouped = df.groupby(['GEOIDCN', 'ACtype'])['Count'].sum().reset_index()

# Calculate The total counts for each 'GEOIDCN'
total_counts = grouped.groupby('GEOIDCN')['Count'].sum().reset_index()
total_counts.rename(columns={'Count': 'TotalCount'}, inplace=True)

# Merge The total counts back to the grouped data
merged = pd.merge(grouped, total_counts, on='GEOIDCN')

# Calculate The percentage
merged['Percentage'] = (merged['Count'] / merged['TotalCount']) * 100

print(merged)


# County boundary
ctboundary = gpd.read_file("<your_path_here>")
non_conus_codes = ['02', '15', '60', '66', '69', '72', '78']
ctboundary = ctboundary[~ctboundary['STATEFP'].isin(non_conus_codes)]

ctboundary.rename(columns={'GEOID': 'GEOIDCN'}, inplace=True)
ctboundary=ctboundary[['GEOIDCN', 'geometry']]

joined = pd.merge(ctboundary,merged, on = ['GEOIDCN'])



import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Assuming 'joined' Is your GeoDataFrame with 'ACtype' and 'Percentage'
ac_types = joined['ACtype'].unique()


fig, axs = plt.subplots(1, len(ac_types), figsize=(15, 6))
cmap = plt.get_cmap('coolwarm_r')  # Specify your colormap
norm = Normalize(vmin=joined['Percentage'].min(), vmax=joined['Percentage'].max())  # Normalize based on data range

for ax, ac_type in zip(axs, ac_types):
    gdf = joined.loc[joined['ACtype'] == ac_type]
    # Plot Each GeoDataFrame subset
    ctboundary.plot(ax=ax, color='gray', linewidth=0.01, edgecolor='lightgray')
    gdf.plot(column='Percentage', cmap=cmap, norm=norm, legend=False, ax=ax, linewidth=0.01, edgecolor='gray')
    ax.set_title(f'{ac_type}',fontsize=20)
    ax.set_axis_off()

# Define The position for the colorbar
# Parameters Are [left, bottom, width, height] in the figure coordinate system
cbar_ax = fig.add_axes([0.15, 0.3, 0.7, 0.03]) # Adjust these values as needed for your layout

# Create A colorbar with the same colormap and normalization
scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
scalar_mappable.set_array([])
cbar = fig.colorbar(scalar_mappable, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Percentage', fontsize=18)  # Set legend label font size

# Adjust The tick label font size
cbar.ax.tick_params(labelsize=15)

# Remove The colorbar border
cbar.outline.set_visible(False)

plt.tight_layout()

fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')

plt.show()





# Urban and rural plot
tracboundary = gpd.read_file("<your_path_here>")

ctboundary = gpd.read_file("<your_path_here>")
ctboundary.rename(columns={'GEOID': 'CBSA_GEOID', 'NAME':"MetroName"}, inplace=True)
ctboundary=ctboundary[['CBSA_GEOID',"MetroName", 'geometry']]

joined = gpd.sjoin( tracboundary,ctboundary) #,  how='inner', predicate='within')
joined.GEOID.unique()


df = pd.read_csv("<your_path_here>")
df['GEOID'] = df['GEOID'].apply(lambda x: f"{x:011d}")


# Create A set of unique GEOIDs that are considered urban
urban_geoids = set(joined['GEOID'].unique())

# Define A function to apply to the GEOID column in tracboundary
def classify_geoid(geoid):
    return 'Urban' if geoid in urban_geoids else 'Rural'

# Apply The function to the GEOID column to create a new 'Area Type' column
tracboundary['urbanicity'] = tracboundary['GEOID'].apply(classify_geoid)

# Tracboundary = pd.DataFrame(tracboundary).drop('geometry', axis = 1)
# DF = Pd.merge(tracboundary, df, on = 'GEOID')
# DF.to_csv("<your_path_here>", Index=False)

# Creating tract level urban
urbandf = tracboundary[['GEOID','urbanicity']]

DF = pd.merge(urbandf, df, on = 'GEOID')





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Make Sure ACtype is a category for ordered plotting
DF['ACtype'] = DF['ACtype'].astype('category')


sns.set_theme(style="ticks", palette="muted")
# Sns.reset_defaults()

plt.figure(figsize=(10, 6))

# Create The box plot
sns.boxplot(
    data=DF,
    x='ACtype',
    y='Percentage',
    hue='urbanicity'
)

# Customizing Axes and legend to match the example
# Add Horizontal legend at the bottom

# Add Horizontal legend at the bottom
plt.legend(
    # Title='Urbanicity',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    fontsize = 13,
    frameon=False
)

plt.xlabel('AC Type', fontsize=15)
plt.ylabel('Percentage', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Plt.title('Distribution of AC Type Percentage by Urbanicity', fontsize=14)

sns.despine(offset=10, trim=True)
plt.tight_layout()

plt.savefig("<your_path_here>", dpi=300, bbox_inches='tight')

plt.show()





# Making summary table
import pandas as pd

# Step 1: Extract State FIPS from GEOIDCN (first 2 digits)
joined['STATEFP'] = joined['GEOIDCN'].str[:2]

# Step 2: Group By state and ACtype to get total count
state_ac_summary = joined.groupby(['STATEFP', 'ACtype'], as_index=False)['Count'].sum()

# Step 3: Get Total housing count per state
state_total = state_ac_summary.groupby('STATEFP', as_index=False)['Count'].sum()
state_total.rename(columns={'Count': 'TotalCount'}, inplace=True)

# Step 4: Merge Total with state-AC summary
state_ac_summary = pd.merge(state_ac_summary, state_total, on='STATEFP')

# Step 5: Calculate Percentage
state_ac_summary['Percentage'] = (state_ac_summary['Count'] / state_ac_summary['TotalCount']) * 100

# Optional: Sort Results
state_ac_summary = state_ac_summary.sort_values(['STATEFP', 'ACtype'])

# View Result
print(state_ac_summary.head())
state_ac_summary.to_csv("<your_path_here>")



