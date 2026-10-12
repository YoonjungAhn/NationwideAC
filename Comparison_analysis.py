# !/usr/bin/env Python3
# -*- Coding: utf-8 -*-
"""
Created on Wed Mar 12 14:14:30 2025

@author: y943a214
"""

import pandas as pd
import geopandas as gpd
import numpy as np


# Making CBSA Data to county level
# Ctboundary = gpd.read_file("<your_path_here>")
# Ctboundary.rename(columns={'GEOID': 'OMB13CBSA', 'NAME':"MetroName"}, inplace=True)

ctboundary = gpd.read_file("<your_path_here>")
ctboundary.rename(columns={'GEOID': 'CBSA_GEOID', 'NAME':"MetroName"}, inplace=True)
ctboundary=ctboundary[['CBSA_GEOID',"MetroName", 'geometry']]

tracboundary = gpd.read_file("<your_path_here>")
tracboundary= tracboundary[['GEOID', 'STATEFP', 'COUNTYFP', 'geometry']]

joined = gpd.sjoin(ctboundary, tracboundary) #,  how='inner', predicate='within')



ahs = pd.read_csv("<your_path_here>")

df1 = pd.read_csv("<your_path_here>")
df2 = pd.read_csv("<your_path_here>")
metroac = pd.read_csv("<your_path_here>")

predict = pd.read_csv("<your_path_here>")
predict['GEOID'] = predict['GEOID'].astype(str).apply(lambda x: x.zfill(11))
predict['RE_ACtype'] = np.where(predict['ACtype'] == 'NoAC', 'NoAC', 'Yes')

total_counts = predict.groupby(['GEOID'])['Count'].sum().reset_index(name='Total_Count')
grouped_data = predict.groupby(['GEOID','RE_ACtype'])['Count'].sum().rename('Count').reset_index()
combined_data = pd.merge(grouped_data, total_counts, on=['GEOID'])
combined_data['Percentage'] = (combined_data['Count'] / combined_data['Total_Count']) * 100



predictdf= pd.merge(joined,combined_data, how='inner', on=['GEOID'])
predictdf = predictdf.groupby(['CBSA_GEOID','RE_ACtype','MetroName'])['Percentage'].mean().reset_index(name='Predict')
predictdf['CBSA_GEOID'] = predictdf['CBSA_GEOID'].astype(str)



# Comparison with AHS
df2 = pd.read_csv("<your_path_here>")
df2.rename(columns={'Count': 'predict_count', 'Percentage':"predict_percentage"}, inplace=True)
df2['OMB13CBSA'] = df2['OMB13CBSA'].astype(str)

DF2 = pd.merge(df2, ctboundary, how='inner', left_on='OMB13CBSA', right_on ='CBSA_GEOID')


ahs = pd.read_csv("<your_path_here>")
ahs.rename(columns={'ACPrimary_Grouped':'New_AC_combine'}, inplace=True)

ahs['OMB13CBSA'] = ahs['OMB13CBSA'].astype(str)

mergedf = pd.merge(DF2, ahs, how='inner', on=['OMB13CBSA','New_AC_combine'])



sns.despine()
plt.rcParams.update({'font.size': 14})  # This sets the default font size for all text

# Assuming Pivot_df is already loaded as shown above
# Filter The data for each type of New_AIRCONDITIONINGrecode and create a scatter plot
filtered_data = mergedf.loc[mergedf.New_AC_combine == "Central"]

threshold = 60# This is an arbitrary threshold for demonstration purposes.

# Create The scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

for index, row in filtered_data.iterrows():
    plt.scatter(row['AHSpercentage'], row['predict_percentage'], color='#2b8cbe', alpha=0.6, s=130)  # Set dot color to black
    difference = abs(row['AHSpercentage'] - row['predict_percentage'])
    if difference > threshold:
        # Jiggle The annotations by adding a small random number to the position
        offset_x = np.random.randint(0, 7)  # Random jitter in the x position
        offset_y = np.random.randint(-15, 6)  # Random jitter in the y position
        # Annotate With a line connecting the dot
        plt.annotate(row['MetroName'],
                     xy=(row['AHSpercentage'], row['predict_percentage']),  # Point of the dot
                     xytext=(row['AHSpercentage'] + offset_x, row['predict_percentage'] + offset_y),  # Offset for text
                     textcoords="data",
                     ha='center',
                     fontsize=10,
                     arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))  # Line properties

# Hide Top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Only Show bottom and left spines (x and y axes)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.xlabel('American Housing Survey (2019-2023)', fontsize=20)
plt.ylabel('Prediction', fontsize=20)
# Plt.title('Scatter Plot for any AC')
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()





sns.despine()
plt.rcParams.update({'font.size': 14})  # This sets the default font size for all text

# Assuming Pivot_df is already loaded as shown above
# Filter The data for each type of New_AIRCONDITIONINGrecode and create a scatter plot
filtered_data = mergedf.loc[mergedf.New_AC_combine == "NoAC"]

threshold = 10# This is an arbitrary threshold for demonstration purposes.

# Create The scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

for index, row in filtered_data.iterrows():
    plt.scatter(row['AHSpercentage'], row['predict_percentage'], color='#2b8cbe', alpha=0.6, s=130)  # Set dot color to black
    difference = abs(row['AHSpercentage'] - row['predict_percentage'])
    if difference > threshold:
        # Jiggle The annotations by adding a small random number to the position
        offset_x = np.random.randint(0, 6)  # Random jitter in the x position
        offset_y = np.random.randint(0, 6)  # Random jitter in the y position
        # Annotate With a line connecting the dot
        plt.annotate(row['MetroName'],
                     xy=(row['AHSpercentage'], row['predict_percentage']),  # Point of the dot
                     xytext=(row['AHSpercentage'] + offset_x, row['predict_percentage'] + offset_y),  # Offset for text
                     textcoords="data",
                     ha='center',
                     fontsize=10,
                     arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))  # Line properties

# Hide Top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Only Show bottom and left spines (x and y axes)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.xlabel('American Housing Survey (2019-2023)', fontsize=20)
plt.ylabel('Prediction', fontsize=20)
# Plt.title('Scatter Plot for any AC')
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=18)




# Combine all the plots of AHS

from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text

import pandas as pd

# Mapping For shorter names
name_mapping = {
    'Baltimore-Columbia-Towson, MD': 'Baltimore, MD',
    'Birmingham-Hoover, AL': 'Birmingham, AL',
    'Cincinnati, OH-KY-IN': 'Cincinnati Metro',
    'Cleveland-Elyria, OH': 'Cleveland, OH',
    'Denver-Aurora-Lakewood, CO': 'Denver, CO',
    'Kansas City, MO-KS': 'Kansas City,MO-KS',
    'Las Vegas-Henderson-Paradise, NV': 'Las Vegas, NV',
    'Memphis, TN-MS-AR': 'Memphis Metro',
    'Milwaukee-Waukesha, WI': 'Milwaukee, WI',
    'Minneapolis-St. Paul-Bloomington, MN-WI': 'Twin Cities, MN-WI',
    'New Orleans-Metairie, LA': 'New Orleans, LA',
    'Oklahoma City, OK': 'Oklahoma City, OK',
    'Pittsburgh, PA': 'Pittsburgh, PA',
    'Portland-Vancouver-Hillsboro, OR-WA': 'Portland Metro',
    'Raleigh-Cary, NC': 'Raleigh, NC',
    'Richmond, VA': 'Richmond, VA',
    'Rochester, NY': 'Rochester, NY',
    'San Antonio-New Braunfels, TX': 'San Antonio, TX',
    'San Jose-Sunnyvale-Santa Clara, CA': 'Silicon Valley, CA',
    'Tampa-St. Petersburg-Clearwater, FL': 'Tampa Bay, FL'
}

# Apply Mapping to create a new column
mergedf['ShortName'] = mergedf['MetroName'].map(name_mapping)

print(filtered_data)


sns.despine()
plt.rcParams.update({'font.size': 14})  # This sets the default font size for all text

# Unique Values for New_AC_combine
unique_ac_types = ['Central', 'Others', 'NoAC']
# Custom Thresholds for each type
thresholds = {'Central': 10, 'NoAC': 5, 'Others': 0}

# Mapping AC Types to more descriptive titles
ac_titles = {
    'Central': 'Central AC',
    'Others': 'Other AC',
    'NoAC': 'No AC'
}

# Create A figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Adjust size as needed

# Loop Through each type and corresponding axis
for ax, ac_type in zip(axes.flatten(), unique_ac_types):
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    texts = []

    for index, row in filtered_data.iterrows():
        ax.scatter(row['AHSpercentage'], row['predict_percentage'], color='#2b8cbe', alpha=0.6, s=130)
        difference = abs(row['AHSpercentage'] - row['predict_percentage'])
        if difference > thresholds[ac_type]:
            texts.append(ax.text(row['AHSpercentage'], row['predict_percentage'], row['ShortName'],
                                 ha='center', fontsize=15))

    # Adjust Texts to minimize overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), ax=ax)

    # Hide Top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Only Show bottom and left spines (x and y axes)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    ax.set_xlabel('American Housing Survey (2019-2023)', fontsize=20)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_title(ac_titles[ac_type], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(False)

plt.tight_layout()
plt.show()



# One way to do it is to make each citieis have differnt colors.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import math

# Unique Values for New_AC_combine
unique_ac_types = ['Central', 'Others', 'NoAC']
# Custom Thresholds for each type
thresholds = {'Central': 10, 'NoAC': 5, 'Others': 10}

# Mapping AC Types to more descriptive titles
ac_titles = {
    'Central': 'Central AC',
    'Others': 'Other AC',
    'NoAC': 'No AC'
}


sns.despine()
plt.rcParams.update({'font.size': 14})  # Sets the default font size for all text

# Create A figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(28, 8))  # Adjust size as needed

# Second Loop to plot data with adjusted axes
for ax, ac_type in zip(axes.flatten(), unique_ac_types):
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    current_max = max( filtered_data['AHSpercentage'].max(), filtered_data['predict_percentage'].max())

    max_value = math.ceil(current_max / 10.0) * 10


    print(max_value)

    texts = []

    for index, row in filtered_data.iterrows():
        ax.scatter(row['AHSpercentage'], row['predict_percentage'], color='#2b8cbe', alpha=0.6, s=130)
        difference = abs(row['AHSpercentage'] - row['predict_percentage'])
        if difference > thresholds[ac_type]:
            texts.append(ax.text(row['AHSpercentage'], row['predict_percentage'], row['ShortName'],
                                 ha='center', fontsize=15))

    adjust_text(texts,
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
            ax=ax,
            expand_points=(2, 2),  # Increase from default values to give more space around points
            expand_text=(1000, 1000),  # Increase to give more space between texts
            force_text=1000,  # Stronger force to push texts apart
            force_points=300,  # Stronger force to push texts away from points
            autoalign='y',  # Align texts vertically to save horizontal space
            only_move={'points':'y', 'text':'y', 'objects':'xy'},  # Allow more flexible movement
            precision=0.1  # Lower precision for faster computation but might be less precise in placement
            )         # Controls the precision of the overlap prevention algorithm


    # Set Uniform axes limits based on the maximum value found
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)

    # Hide Top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Only Show bottom and left spines (x and y axes)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Set Axis labels and title
    ax.set_xlabel('American Housing Survey (2019-2023)', fontsize=20)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_title(ac_titles[ac_type], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(False)

plt.tight_layout()

plt.show()






# Making legend as city names

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math


# Unique Values for New_AC_combine
unique_ac_types = ['Central', 'Others', 'NoAC']
# Custom Thresholds for each type
thresholds = {'Central': 10, 'NoAC': 5, 'Others': 10}

# Mapping AC Types to more descriptive titles
ac_titles = {
    'Central': 'Central AC',
    'Others': 'Other AC',
    'NoAC': 'No AC'
}


sns.despine()
plt.rcParams.update({'font.size': 14})  # Sets the default font size for all text

# Create A figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(20, 11))  # Adjust size as needed

# Change The color palette here (e.g., 'plasma', 'inferno', 'magma', 'cividis')
colors = plt.cm.inferno(np.linspace(0, 1, len(mergedf['ShortName'].unique()))) #inferno, magma, or cividis.
color_map = dict(zip(mergedf['ShortName'].unique(), colors))

# Plot Data
for ax, ac_type in zip(axes.flatten(), unique_ac_types):
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    current_max = max(filtered_data['AHSpercentage'].max(), filtered_data['predict_percentage'].max())
    max_value = (math.ceil(current_max / 10.0) * 10)+5

    for index, row in filtered_data.iterrows():
        color = color_map[row['ShortName']]
        ax.scatter(row['AHSpercentage'], row['predict_percentage'], color=color, marker='o', alpha=0.6, s=130)

    # Set Axis limits
    ax.set_xlim(-2, max_value)
    ax.set_ylim(-2, max_value)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('American Housing Survey (2019-2023)', fontsize=20)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_title(ac_titles[ac_type], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(False)

# Add A legend without a border and organized into 3 rows and 5 columns
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=name)
           for name, color in color_map.items()]
legend = axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=False, shadow=False, ncol=4, title="City Names", fontsize=23, title_fontsize=25)
legend.get_frame().set_linewidth(0.0)


plt.subplots_adjust(wspace=-1.2)  # Adjust the space between plots; decrease to bring them closer
plt.tight_layout()
fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')

plt.show()

import pandas as pd
import scipy.stats as stats

# Assuming Mergedf is your main DataFrame
# Ensure That unique_ac_types is defined, e.g.:
unique_ac_types = mergedf['New_AC_combine'].unique()

# Initialize A list to hold the results
results = []

# Loop Through each AC type and calculate correlation coefficient and p-value
for ac_type in unique_ac_types:
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    corr_coef, p_value = stats.pearsonr(filtered_data['AHSpercentage'], filtered_data['predict_percentage'])
    results.append({
        'AC Type': ac_type,
        'Correlation Coefficient': corr_coef,
        'P-Value': p_value
    })

# Convert The results list to a DataFrame for easier viewing
results_df = pd.DataFrame(results)

# Print The results DataFrame
print(results_df)




# National AHS Level all of them
ahs = pd.read_csv("<your_path_here>")

# Create A mapping dictionary for division codes to names
division_mapping = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific"
}
# Replace The 'DIVISON' column values using the mapping dictionary
ahs['DIVISION'] = ahs['DIVISION'].replace(division_mapping)



df3 = pd.read_csv("<your_path_here>")

df3['GEOID'] = df3['GEOID'].astype(str).apply(lambda x: x.zfill(11))
df3['StateFIPS'] = df3['GEOID'].str[0:2]
df3['New_AC_combine']=df3.ACtype.replace('Evaporative Cooler', 'Others')

division =pd.read_csv("<your_path_here>")
division['StateFIPS'] = division['StateFIPS'].astype(str).apply(lambda x: x.zfill(2))

divisiondf = pd.merge(df3,division, on ='StateFIPS' )


grouped_data = divisiondf.groupby('DIVISION')['New_AC_combine'].value_counts().rename('Count')
grouped_percentage = divisiondf.groupby('DIVISION')['New_AC_combine'].value_counts(normalize=True).rename('Percentage') * 100

combined_data = pd.DataFrame({
    'Count': grouped_data,
    'predict_percentage': grouped_percentage
}).reset_index()

print(combined_data)

mergedf= pd.merge(ahs,combined_data, left_on =['DIVISION','ACPrimary_Grouped'] ,right_on = ['DIVISION','New_AC_combine'])



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

sns.despine()
plt.rcParams.update({'font.size': 14})  # Sets default font size

# Create Figure
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

# Sort DIVISION Names alphabetically
sorted_divisions = sorted(mergedf['DIVISION'].unique())

# Create A color map using the sorted divisions
colors = plt.cm.inferno(np.linspace(0, 1, len(sorted_divisions)))
color_map = dict(zip(sorted_divisions, colors))

unique_ac_types = ['Central', 'Others', 'NoAC']

# Plot Each subplot
for ax, ac_type in zip(axes.flatten(), unique_ac_types):
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    current_max = max(filtered_data['AHSpercentage'].max(), filtered_data['predict_percentage'].max())
    max_value = (math.ceil(current_max / 10.0) * 10) + 5

    for index, row in filtered_data.iterrows():
        color = color_map[row['DIVISION']]
        ax.scatter(row['AHSpercentage'], row['predict_percentage'], color=color, marker='o', alpha=0.6, s=130)

    ax.set_xlim(-2, max_value)
    ax.set_ylim(-2, max_value)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('American Housing Survey (2019–2023)', fontsize=20)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_title(ac_titles[ac_type], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(False)

# Create Sorted legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[name], markersize=10, label=name)
           for name in sorted_divisions]
legend = axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25),
                        fancybox=False, shadow=False, ncol=3, title="Division",
                        fontsize=23, title_fontsize=25)
legend.get_frame().set_linewidth(0.0)

plt.subplots_adjust(wspace=-1)
plt.tight_layout()
fig.savefig("<your_path_here>",
            dpi=300, bbox_inches='tight')
plt.show()




import pandas as pd
import scipy.stats as stats

# Assuming Mergedf is your main DataFrame
# Ensure That unique_ac_types is defined, e.g.:
unique_ac_types = mergedf['New_AC_combine'].unique()

# Initialize A list to hold the results
results = []

# Loop Through each AC type and calculate correlation coefficient and p-value
for ac_type in unique_ac_types:
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    corr_coef, p_value = stats.pearsonr(filtered_data['AHSpercentage'], filtered_data['predict_percentage'])
    results.append({
        'AC Type': ac_type,
        'Correlation Coefficient': corr_coef,
        'P-Value': p_value
    })

# Convert The results list to a DataFrame for easier viewing
results_df = pd.DataFrame(results)

# Print The results DataFrame
print(results_df)







# National AHS Level all of no metro level
ctboundary = gpd.read_file("<your_path_here>")
ctboundary = gpd.read_file("<your_path_here>")
ctboundary.rename(columns={'GEOID': 'CBSA_GEOID', 'NAME':"MetroName"}, inplace=True)
ctboundary=ctboundary[['CBSA_GEOID',"MetroName", 'geometry']]


tracboundary = gpd.read_file("<your_path_here>")
tracboundary= tracboundary[['GEOID', 'STATEFP', 'COUNTYFP', 'geometry']]

joined = gpd.sjoin(ctboundary, tracboundary) #,  how='inner', predicate='within')
joineddf = pd.DataFrame(joined[[ 'GEOID','CBSA_GEOID', 'MetroName','STATEFP']])
joineddf['GEOIDCN']= joineddf.GEOID.str[0:5]
joineddf.GEOIDCN.unique()

predict = pd.read_csv("<your_path_here>")
predict['GEOID'] = predict['GEOID'].astype(str).apply(lambda x: x.zfill(11))
predict['GEOIDCN'] =predict.GEOID.str[0:5]
predict['New_AC_combine']=predict.ACtype.replace('Evaporative Cooler', 'Others')


# Getting Unique GEOIDCN from joineddf
unique_geo_ids = joineddf['GEOIDCN'].unique()

# Filtering Predict to keep rows where GEOIDCN is not in the unique GEOIDCN from joineddf
filtered_predict = predict[~predict['GEOIDCN'].isin(unique_geo_ids)]
filtered_predict['StateFIPS'] =filtered_predict.GEOID.str[0:2]


division =pd.read_csv("<your_path_here>")
division['StateFIPS'] = division['StateFIPS'].astype(str).apply(lambda x: x.zfill(2))

divisiondf = pd.merge(filtered_predict,division, on ='StateFIPS', how='left' )


grouped_data = divisiondf.groupby('DIVISION')['New_AC_combine'].value_counts().rename('Count')
grouped_percentage = divisiondf.groupby('DIVISION')['New_AC_combine'].value_counts(normalize=True).rename('Percentage') * 100

combined_data = pd.DataFrame({
    'Count': grouped_data,
    'predict_percentage': grouped_percentage
}).reset_index()


ahs = pd.read_csv("<your_path_here>")

# Create A mapping dictionary for division codes to names
division_mapping = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific"
}
# Replace The 'DIVISON' column values using the mapping dictionary
ahs['DIVISION'] = ahs['DIVISION'].replace(division_mapping)



mergedf= pd.merge(ahs,combined_data, left_on =['DIVISION','ACPrimary_Grouped'] ,right_on = ['DIVISION','New_AC_combine'])



sns.despine()
plt.rcParams.update({'font.size': 14})  # Sets default font size

# Create Figure
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

# Sort DIVISION Names alphabetically
sorted_divisions = sorted(mergedf['DIVISION'].unique())

# Create A color map using the sorted divisions
colors = plt.cm.inferno(np.linspace(0, 1, len(sorted_divisions)))
color_map = dict(zip(sorted_divisions, colors))

unique_ac_types = ['Central', 'Others', 'NoAC']



# Plot Data
for ax, ac_type in zip(axes.flatten(), unique_ac_types):
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    current_max = max(filtered_data['AHSpercentage'].max(), filtered_data['predict_percentage'].max())
    max_value = (math.ceil(current_max / 10.0) * 10)+5

    for index, row in filtered_data.iterrows():
        color = color_map[row['DIVISION']]
        ax.scatter(row['AHSpercentage'], row['predict_percentage'], color=color, marker='o', alpha=0.6, s=130)

    # Set Axis limits
    ax.set_xlim(-2, max_value)
    ax.set_ylim(-2, max_value)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('American Housing Survey (2019-2023)', fontsize=20)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_title(ac_titles[ac_type], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(False)


# Create Sorted legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[name], markersize=10, label=name)
           for name in sorted_divisions]
legend = axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25),
                        fancybox=False, shadow=False, ncol=3, title="Division",
                        fontsize=23, title_fontsize=25)
legend.get_frame().set_linewidth(0.0)

plt.subplots_adjust(wspace=-1)
plt.tight_layout()

fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')

plt.show()



import pandas as pd
import scipy.stats as stats

# Assuming Mergedf is your main DataFrame
# Ensure That unique_ac_types is defined, e.g.:
unique_ac_types = mergedf['New_AC_combine'].unique()

# Initialize A list to hold the results
results = []

# Loop Through each AC type and calculate correlation coefficient and p-value
for ac_type in unique_ac_types:
    filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]
    corr_coef, p_value = stats.pearsonr(filtered_data['AHSpercentage'], filtered_data['predict_percentage'])
    results.append({
        'AC Type': ac_type,
        'Correlation Coefficient': corr_coef,
        'P-Value': p_value
    })

# Convert The results list to a DataFrame for easier viewing
results_df = pd.DataFrame(results)

# Print The results DataFrame
print(results_df)





# Comparison With SARA et al
import pandas as pd

ctboundary = gpd.read_file("<your_path_here>")
ctboundary.rename(columns={'GEOID': 'CBSA_GEOID', 'NAME':"MetroName"}, inplace=True)
ctboundary=ctboundary[['CBSA_GEOID',"MetroName", 'geometry']]

tracboundary = gpd.read_file("<your_path_here>")
tracboundary= tracboundary[['GEOID', 'STATEFP', 'COUNTYFP', 'geometry']]

joined = gpd.sjoin(ctboundary, tracboundary) #,  how='inner', predicate='within')
joined = joined[['GEOID', 'CBSA_GEOID',"MetroName",]]


# Assuming 'ctboundary' And 'table2' are already defined dataframes
saradf = pd.read_csv("<your_path_here>")

# Step 3: Create A 'MetroName' in table2 by combining 'City' and 'State' columns
saradf['MetroName'] = saradf['City'].str.strip().str.title() + ', ' + saradf['State'].str.strip()


mergeddf = pd.merge(ctboundary, saradf, on='MetroName', how='right')
mergeddf.drop(columns=['geometry'], inplace=True)


predict = pd.read_csv("<your_path_here>")
predict['GEOID'] = predict['GEOID'].astype(str).apply(lambda x: x.zfill(11))
total_counts = predict.groupby(['GEOID'])['Count'].sum().reset_index(name='Total_Count')
grouped_data = predict.groupby(['GEOID','ACtype'])['Count'].sum().rename('predict_count').reset_index()
combined_data = pd.merge(grouped_data, total_counts, on=['GEOID'])
combined_data['predict_percentage'] = (combined_data['predict_count'] / combined_data['Total_Count']) * 100


predictdf= pd.merge(joined,combined_data, how='inner', on=['GEOID'])
predictdf = predictdf.groupby(['CBSA_GEOID','ACtype','MetroName'])['predict_percentage'].mean().reset_index(name='predict_percentage')
predictdf['CBSA_GEOID'] = predictdf['CBSA_GEOID'].astype(str)

merged_df = pd.merge(predictdf, mergeddf , on = ['CBSA_GEOID','MetroName']  , how='inner') #.drop(columns=['geometry'], inplace=True)

# Assuming Filtered_data is already loaded
filtered_data = merged_df.loc[merged_df.ACtype == "Central"]
threshold = 33 # This is an arbitrary threshold for demonstration purposes.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Create The scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
texts = []  # List to hold the adjust_text objects

for index, row in filtered_data.iterrows():
    # Plot Each point
    plt.scatter(row['Average AC prevalence'], row['predict_percentage'], color='#2b8cbe', alpha=0.6, s=130)

    difference = abs(row['Average AC prevalence'] - row['predict_percentage'])
    if difference > threshold:
        # Create An annotation for points above the threshold
        text = plt.text(row['Average AC prevalence'], row['predict_percentage'], row['MetroName'],
                        ha='center', fontsize=10)
        texts.append(text)

# Use Adjust_text to dynamically adjust annotations, with arrow properties
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

# Hide Top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Only Show bottom and left spines (x and y axes)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.xlabel('Sara et al. (2020)', fontsize=20)
plt.ylabel('Prediction', fontsize=20)
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=18)

fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')
plt.show()




# Comparison With Romitti et al
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

predict = pd.read_csv("<your_path_here>")
predict['GEOID'] = predict['GEOID'].astype(str).apply(lambda x: x.zfill(11))
predict['RE_ACtype'] = np.where(predict['ACtype'] == 'NoAC', 'NoAC', 'Yes')

total_counts = predict.groupby(['GEOID'])['Count'].sum().reset_index(name='Total_Count')
grouped_data = predict.groupby(['GEOID','RE_ACtype'])['Count'].sum().rename('Count').reset_index()
combined_data = pd.merge(grouped_data, total_counts, on=['GEOID'])
combined_data['Percentage'] = (combined_data['Count'] / combined_data['Total_Count']) * 100



predictdf= pd.merge(joined,combined_data, how='inner', on=['GEOID'])
predictdf = predictdf.groupby(['CBSA_GEOID','RE_ACtype','MetroName'])['Percentage'].mean().reset_index(name='Predict')
predictdf['CBSA_GEOID'] = predictdf['CBSA_GEOID'].astype(str)



metroac = pd.read_csv("<your_path_here>")
metroac['Romitti_Percentage'] = metroac['ac_prob']*100
# Metroac.rename(columns={'CBSA_GEOID': 'OMB13CBSA'}, inplace=True)
metroac['TRACT_GEOID'] = metroac['TRACT_GEOID'].astype(str).apply(lambda x: x.zfill(11))

metroacdf = metroac.groupby('CBSA_GEOID')['Romitti_Percentage'].mean().reset_index(name='Romitti_Percentage')
metroacdf['CBSA_GEOID'] = metroacdf['CBSA_GEOID'].astype(str)

mergedf = pd.merge(predictdf, metroacdf, how='inner', on='CBSA_GEOID')
# Mergedf = pd.merge(combined_data, metroac, how='inner', left_on='GEOID', right_on = 'TRACT_GEOID')



# Set The style of seaborn and matplotlib parameters
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import numpy as np

# Set The style of seaborn and matplotlib parameters
sns.set_style("white")  # Changed from "whitegrid" to better see the annotations
sns.despine()
plt.rcParams.update({'font.size': 14})  # This sets the default font size for all text

# Assuming Filtered_data is already loaded
filtered_data = mergedf.loc[mergedf.RE_ACtype != "NoAC"]
threshold = 25  # This is an arbitrary threshold for demonstration purposes.

# Create The scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
texts = []

for index, row in filtered_data.iterrows():
    # Plot Each point
    plt.scatter(row['Romitti_Percentage'], row['Predict'], color='#2b8cbe', alpha=0.6, s=130)

    difference = abs(row['Romitti_Percentage'] - row['Predict'])
    if difference > threshold:
        # Create An annotation for points above the threshold
        text = plt.text(row['Romitti_Percentage'], row['Predict'], row['MetroName'],
                        ha='center', fontsize=10)
        texts.append(text)

# Use Adjust_text to dynamically adjust annotations, with arrow properties
adjust_text(texts, expand_points=(1.2, 1.5), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

# Hide Top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Only Show bottom and left spines (x and y axes)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.xlabel('Romitti et al. (2022)', fontsize=20)
plt.ylabel('Prediction', fontsize=20)
plt.grid(False)
plt.tick_params(axis='both', which='major', labelsize=18)

fig.savefig("<your_path_here>", dpi=300, bbox_inches='tight')
plt.show()





