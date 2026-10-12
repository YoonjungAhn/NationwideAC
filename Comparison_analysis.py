#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:14:30 2025

@author: y943a214

Validation: compare the predicted AC prevalence with other data sources.

  1. American Housing Survey (AHS), by metro area
  2. AHS, national, by census division (all tracts)
  3. AHS, national, by census division (non-metro tracts only)
  4. Sera et al. (2020), by metro area
  5. Romitti et al. (2022), by metro area
  6. New York City Housing and Vacancy Survey 2023 (data preparation only)

Each comparison draws a scatter plot (other source on x, prediction on y) and
prints the Pearson correlation for each AC type.

Inputs (set DATA_DIR and FIG_DIR below)
---------------------------------------
results/summary_tract_2020.csv     tract-level AC counts by type, output of ACprob_to_county_zip.py
boundary/...                       2020 CBSA (metro area) and 2020 census tract shapefiles (NHGIS)
comparisondata/...                 comparison data sets (see PATHS below)
AHS/Labels/Census_Bureau_Regions_Divisions.csv   state FIPS -> census division
"""

import math
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from adjustText import adjust_text


# =============================================================================
# 0. Settings: file locations
# =============================================================================
DATA_DIR = '<your_path_here>'  # project "Data" folder
FIG_DIR = '<your_path_here>'   # folder for the saved figures

PATHS = {
    # Boundaries (NHGIS; replace "nhgis0075" with your own extract number)
    'cbsa_2020': os.path.join(DATA_DIR, 'NHGIS/2020_us_cbsa/US_cbsa_2020.shp'),
    'tract_2020': os.path.join(DATA_DIR, 'boundary/nhgis0075_shapefile_tl2020_us_tract_2020/US_tract_2020.shp'),
    # Predictions: tract-level AC counts by type (output of ACprob_to_county_zip.py)
    'predict_tract': os.path.join(DATA_DIR, 'results/summary_tract_2020.csv'),
    # Predictions summarized by metro area (OMB13CBSA, New_AC_combine, Count, Percentage)
    'predict_metro': os.path.join(DATA_DIR, 'comparisondata/AC_prediction_metro_central_no_others.csv'),
    # American Housing Survey (https://www.census.gov/programs-surveys/ahs.html), share of homes by AC type
    'ahs_metro': os.path.join(DATA_DIR, 'comparisondata/AHS_2019_2021.csv'),
    'ahs_national': os.path.join(DATA_DIR, 'comparisondata/AHS_national_2021.csv'),
    'ahs_national_nometro': os.path.join(DATA_DIR, 'comparisondata/AHS_national_noMetro_2021.csv'),
    'divisions': os.path.join(DATA_DIR, 'AHS/Labels/Census_Bureau_Regions_Divisions.csv'),
    # Sera et al. (2020): average AC prevalence by city
    'sera': os.path.join(DATA_DIR, 'comparisondata/sara.csv'),
    # Romitti et al. (2022): AC probability by tract and metro area
    'romitti': os.path.join(DATA_DIR, 'comparisondata/US_metro_ac_prob-1.csv'),
    # NYC Housing and Vacancy Survey 2023 public use files
    'nyc_allunits': os.path.join(DATA_DIR, 'comparisondata/nyc data/allunits_puf_23.csv'),
    'nyc_occupied': os.path.join(DATA_DIR, 'comparisondata/nyc data/occupied_puf_23.csv'),
}

# Plot titles for each AC type
AC_TITLES = {'Central': 'Central AC', 'Others': 'Other AC', 'NoAC': 'No AC'}
AC_TYPES = ['Central', 'Others', 'NoAC']

# AHS census division codes -> names
DIVISION_MAPPING = {
    1: "New England", 2: "Middle Atlantic", 3: "East North Central",
    4: "West North Central", 5: "South Atlantic", 6: "East South Central",
    7: "West South Central", 8: "Mountain", 9: "Pacific"
}

plt.rcParams.update({'font.size': 14})


# =============================================================================
# Helper functions
# =============================================================================
def load_cbsa_tract_join():
    """Return the 2020 CBSA boundaries and every 2020 tract that intersects a CBSA."""
    ctboundary = gpd.read_file(PATHS['cbsa_2020'])
    ctboundary.rename(columns={'GEOID': 'CBSA_GEOID', 'NAME': "MetroName"}, inplace=True)
    ctboundary = ctboundary[['CBSA_GEOID', "MetroName", 'geometry']]

    tracboundary = gpd.read_file(PATHS['tract_2020'])
    tracboundary = tracboundary[['GEOID', 'STATEFP', 'COUNTYFP', 'geometry']]

    joined = gpd.sjoin(ctboundary, tracboundary)
    return ctboundary, joined


def load_tract_predictions():
    """Tract-level predicted AC counts with an 11-digit GEOID."""
    predict = pd.read_csv(PATHS['predict_tract'])
    predict['GEOID'] = predict['GEOID'].astype(str).apply(lambda x: x.zfill(11))
    return predict


def load_divisions():
    """State FIPS (2 digits) -> census division."""
    division = pd.read_csv(PATHS['divisions'])
    division['StateFIPS'] = division['StateFIPS'].astype(str).apply(lambda x: x.zfill(2))
    return division


def load_ahs_national(path):
    """AHS national table with division names instead of codes."""
    ahs = pd.read_csv(path)
    ahs['DIVISION'] = ahs['DIVISION'].replace(DIVISION_MAPPING)
    return ahs


def percent_by_division(tracts):
    """Share of each AC type (Evaporative Cooler counted as Others) by census division."""
    grouped_data = tracts.groupby('DIVISION')['New_AC_combine'].value_counts().rename('Count')
    grouped_percentage = tracts.groupby('DIVISION')['New_AC_combine'].value_counts(normalize=True).rename('Percentage') * 100
    return pd.DataFrame({'Count': grouped_data, 'predict_percentage': grouped_percentage}).reset_index()


def correlation_table(mergedf, x='AHSpercentage', y='predict_percentage', group='New_AC_combine'):
    """Pearson correlation and p-value between the two sources, for each AC type."""
    results = []
    for ac_type in mergedf[group].unique():
        filtered_data = mergedf[mergedf[group] == ac_type]
        corr_coef, p_value = stats.pearsonr(filtered_data[x], filtered_data[y])
        results.append({'AC Type': ac_type, 'Correlation Coefficient': corr_coef, 'P-Value': p_value})
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


def plot_ahs_panels(mergedf, color_col, color_names, legend_title, ncol, figsize, wspace, outfile):
    """Three scatter plots (Central, Other, No AC): AHS on x, prediction on y, colored by color_col."""
    sns.despine()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = plt.cm.inferno(np.linspace(0, 1, len(color_names)))
    color_map = dict(zip(color_names, colors))

    for ax, ac_type in zip(axes.flatten(), AC_TYPES):
        filtered_data = mergedf[mergedf['New_AC_combine'] == ac_type]

        # Same x and y range, rounded up to the next 10 (+5 margin)
        current_max = max(filtered_data['AHSpercentage'].max(), filtered_data['predict_percentage'].max())
        max_value = (math.ceil(current_max / 10.0) * 10) + 5

        for _, row in filtered_data.iterrows():
            ax.scatter(row['AHSpercentage'], row['predict_percentage'],
                       color=color_map[row[color_col]], marker='o', alpha=0.6, s=130)

        ax.set_xlim(-2, max_value)
        ax.set_ylim(-2, max_value)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('American Housing Survey (2019-2023)', fontsize=20)
        ax.set_ylabel('Prediction', fontsize=20)
        ax.set_title(AC_TITLES[ac_type], fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(False)

    # One legend below the middle panel
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[name], markersize=10, label=name)
               for name in color_names]
    legend = axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25),
                            fancybox=False, shadow=False, ncol=ncol, title=legend_title,
                            fontsize=23, title_fontsize=25)
    legend.get_frame().set_linewidth(0.0)

    plt.subplots_adjust(wspace=wspace)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.show()


def plot_labeled_scatter(data, x, y, threshold, xlabel, outfile, **adjust_kwargs):
    """Scatter plot of two sources; metro areas that differ by more than `threshold` are labeled."""
    sns.set_style("white")
    sns.despine()
    fig, ax = plt.subplots(figsize=(8, 6))
    texts = []

    for _, row in data.iterrows():
        plt.scatter(row[x], row[y], color='#2b8cbe', alpha=0.6, s=130)
        if abs(row[x] - row[y]) > threshold:
            texts.append(plt.text(row[x], row[y], row['MetroName'], ha='center', fontsize=10))

    # Move labels so they do not overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), **adjust_kwargs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=18)

    fig.savefig(os.path.join(FIG_DIR, outfile), dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# Shared data
# =============================================================================
ctboundary, joined = load_cbsa_tract_join()   # metro areas and the tracts inside them
divisions = load_divisions()


# =============================================================================
# 1. AHS by metro area
# =============================================================================
# Predicted share of each AC type by metro area
df2 = pd.read_csv(PATHS['predict_metro'])
df2.rename(columns={'Count': 'predict_count', 'Percentage': "predict_percentage"}, inplace=True)
df2['OMB13CBSA'] = df2['OMB13CBSA'].astype(str)
DF2 = pd.merge(df2, ctboundary, how='inner', left_on='OMB13CBSA', right_on='CBSA_GEOID')

# AHS share of each AC type by metro area
ahs = pd.read_csv(PATHS['ahs_metro'])
ahs.rename(columns={'ACPrimary_Grouped': 'New_AC_combine'}, inplace=True)
ahs['OMB13CBSA'] = ahs['OMB13CBSA'].astype(str)

mergedf = pd.merge(DF2, ahs, how='inner', on=['OMB13CBSA', 'New_AC_combine'])

# Shorter metro names for the legend
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
mergedf['ShortName'] = mergedf['MetroName'].map(name_mapping)

plot_ahs_panels(mergedf, color_col='ShortName', color_names=list(mergedf['ShortName'].unique()),
                legend_title="City Names", ncol=4, figsize=(20, 11), wspace=-1.2,
                outfile='Comparison_AHS_metro.png')
correlation_table(mergedf)


# =============================================================================
# 2. AHS national, by census division (all tracts)
# =============================================================================
df3 = load_tract_predictions()
df3['StateFIPS'] = df3['GEOID'].str[0:2]
df3['New_AC_combine'] = df3.ACtype.replace('Evaporative Cooler', 'Others')
divisiondf = pd.merge(df3, divisions, on='StateFIPS')

combined_data = percent_by_division(divisiondf)
print(combined_data)

ahs = load_ahs_national(PATHS['ahs_national'])
mergedf = pd.merge(ahs, combined_data, left_on=['DIVISION', 'ACPrimary_Grouped'], right_on=['DIVISION', 'New_AC_combine'])

plot_ahs_panels(mergedf, color_col='DIVISION', color_names=sorted(mergedf['DIVISION'].unique()),
                legend_title="Division", ncol=3, figsize=(20, 10), wspace=-1,
                outfile='Comparison_AHS_national_all.png')
correlation_table(mergedf)


# =============================================================================
# 3. AHS national, by census division (non-metro tracts only)
# =============================================================================
# Counties that contain at least one metro-area tract
joineddf = pd.DataFrame(joined[['GEOID', 'CBSA_GEOID', 'MetroName', 'STATEFP']])
joineddf['GEOIDCN'] = joineddf.GEOID.str[0:5]
metro_counties = joineddf['GEOIDCN'].unique()

# Keep tracts in counties outside every metro area
predict = load_tract_predictions()
predict['GEOIDCN'] = predict.GEOID.str[0:5]
predict['New_AC_combine'] = predict.ACtype.replace('Evaporative Cooler', 'Others')
filtered_predict = predict[~predict['GEOIDCN'].isin(metro_counties)].copy()
filtered_predict['StateFIPS'] = filtered_predict.GEOID.str[0:2]

divisiondf = pd.merge(filtered_predict, divisions, on='StateFIPS', how='left')
combined_data = percent_by_division(divisiondf)

ahs = load_ahs_national(PATHS['ahs_national_nometro'])
mergedf = pd.merge(ahs, combined_data, left_on=['DIVISION', 'ACPrimary_Grouped'], right_on=['DIVISION', 'New_AC_combine'])

plot_ahs_panels(mergedf, color_col='DIVISION', color_names=sorted(mergedf['DIVISION'].unique()),
                legend_title="Division", ncol=3, figsize=(20, 10), wspace=-1,
                outfile='Comparison_AHS_national_non_metro.png')
correlation_table(mergedf)


# =============================================================================
# 4. Sera et al. (2020), by metro area (Central AC)
# =============================================================================
joined_ids = joined[['GEOID', 'CBSA_GEOID', "MetroName"]]

# Match Sera et al. cities to CBSA names ("City, ST")
saradf = pd.read_csv(PATHS['sera'])
saradf['MetroName'] = saradf['City'].str.strip().str.title() + ', ' + saradf['State'].str.strip()
mergeddf = pd.merge(ctboundary, saradf, on='MetroName', how='right').drop(columns=['geometry'])

# Predicted share of each AC type by tract, then averaged over the tracts of each metro area
predict = load_tract_predictions()
total_counts = predict.groupby(['GEOID'])['Count'].sum().reset_index(name='Total_Count')
grouped_data = predict.groupby(['GEOID', 'ACtype'])['Count'].sum().rename('predict_count').reset_index()
combined_data = pd.merge(grouped_data, total_counts, on=['GEOID'])
combined_data['predict_percentage'] = (combined_data['predict_count'] / combined_data['Total_Count']) * 100

predictdf = pd.merge(joined_ids, combined_data, how='inner', on=['GEOID'])
predictdf = predictdf.groupby(['CBSA_GEOID', 'ACtype', 'MetroName'])['predict_percentage'].mean().reset_index()
predictdf['CBSA_GEOID'] = predictdf['CBSA_GEOID'].astype(str)

merged_df = pd.merge(predictdf, mergeddf, on=['CBSA_GEOID', 'MetroName'], how='inner')

plot_labeled_scatter(merged_df.loc[merged_df.ACtype == "Central"],
                     x='Average AC prevalence', y='predict_percentage', threshold=33,
                     xlabel='Sera et al. (2020)', outfile='Comparison_Sara.png')


# =============================================================================
# 5. Romitti et al. (2022), by metro area (any AC)
# =============================================================================
# Predicted share of homes with any AC (every type except NoAC) by tract, averaged by metro area
predict = load_tract_predictions()
predict['RE_ACtype'] = np.where(predict['ACtype'] == 'NoAC', 'NoAC', 'Yes')

total_counts = predict.groupby(['GEOID'])['Count'].sum().reset_index(name='Total_Count')
grouped_data = predict.groupby(['GEOID', 'RE_ACtype'])['Count'].sum().rename('Count').reset_index()
combined_data = pd.merge(grouped_data, total_counts, on=['GEOID'])
combined_data['Percentage'] = (combined_data['Count'] / combined_data['Total_Count']) * 100

predictdf = pd.merge(joined_ids, combined_data, how='inner', on=['GEOID'])
predictdf = predictdf.groupby(['CBSA_GEOID', 'RE_ACtype', 'MetroName'])['Percentage'].mean().reset_index(name='Predict')
predictdf['CBSA_GEOID'] = predictdf['CBSA_GEOID'].astype(str)

# Romitti et al. AC probability (0-1) -> percentage, averaged by metro area
metroac = pd.read_csv(PATHS['romitti'])
metroac['Romitti_Percentage'] = metroac['ac_prob'] * 100
metroac['TRACT_GEOID'] = metroac['TRACT_GEOID'].astype(str).apply(lambda x: x.zfill(11))
metroacdf = metroac.groupby('CBSA_GEOID')['Romitti_Percentage'].mean().reset_index(name='Romitti_Percentage')
metroacdf['CBSA_GEOID'] = metroacdf['CBSA_GEOID'].astype(str)

mergedf = pd.merge(predictdf, metroacdf, how='inner', on='CBSA_GEOID')

plot_labeled_scatter(mergedf.loc[mergedf.RE_ACtype != "NoAC"],
                     x='Romitti_Percentage', y='Predict', threshold=25,
                     xlabel='Romitti et al. (2022)', outfile='Comparison_Romiiti.png',
                     expand_points=(1.2, 1.5))


# =============================================================================
# 6. NYC Housing and Vacancy Survey 2023 (data preparation only)
# =============================================================================
# 6a. All units: central AC (CENTRALAIR) by borough
nyc = pd.read_csv(PATHS['nyc_allunits'])[['CENTRALAIR', 'BORO']]
nyc['BORO'] = nyc['BORO'].astype(str)

# Borough code -> county FIPS
nyc_fips_df = pd.DataFrame({
    'BORO': ['1', '2', '3', '4', '5'],
    'Borough': ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
    'County': ['Bronx County', 'Kings County', 'New York County', 'Queens County', 'Richmond County'],
    'GEOIDCNT': ['36005', '36047', '36061', '36081', '36085']
})
nycdf = pd.merge(nyc, nyc_fips_df, on=['BORO'])
nycdf['ACtype'] = nycdf['CENTRALAIR'].map({-1: 'NA', 1: 'Central', 2: 'NotCentral'})

# Share of each AC type by county (borough)
grouped_data = nycdf.groupby('GEOIDCNT')['ACtype'].value_counts().rename('Count').reset_index()
grouped_percentage = (nycdf.groupby('GEOIDCNT')['ACtype'].value_counts(normalize=True).rename('Percentage') * 100).reset_index()
NYC = pd.merge(grouped_data, grouped_percentage, on=['GEOIDCNT', 'ACtype'])

# 6b. Occupied units: AC unit type (ACUNIT), citywide share
nyc_occupied = pd.read_csv(PATHS['nyc_occupied'])[['ACUNIT']]
nyc_occupied['ACtype'] = nyc_occupied['ACUNIT'].map({-3: 'NA', -1: "not reported", 1: 'Others', 2: 'Central'})
nyc_occupied = nyc_occupied.groupby('ACtype').value_counts().rename('Count').reset_index()
nyc_occupied['percentage'] = nyc_occupied.Count / nyc_occupied.Count.sum()

# 6c. Predicted share of each AC type in the five NYC counties
predict = load_tract_predictions()
predict['GEOIDCNT'] = predict['GEOID'].str[0:5]
total_counts = predict.groupby(['GEOIDCNT'])['Count'].sum().reset_index(name='Total_Count')
grouped_data = predict.groupby(['GEOIDCNT', 'ACtype'])['Count'].sum().rename('predict_count').reset_index()
combined_data = pd.merge(grouped_data, total_counts, on=['GEOIDCNT'])
combined_data['predict_percentage'] = (combined_data['predict_count'] / combined_data['Total_Count']) * 100

nyc_predicted = combined_data[combined_data['GEOIDCNT'].isin(nyc_fips_df.GEOIDCNT.unique())]
print(nyc_predicted)
