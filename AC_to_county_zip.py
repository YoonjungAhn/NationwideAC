# !/usr/bin/env Python3
# -*- Coding: utf-8 -*-
"""
Created on Thu Feb 20 09:47:28 2025

@author: y943a214
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np

# Trcatbd = gpd.read_file("<your_path_here>")

# Trcatbd = gpd.read_file("<your_path_here>")
# Set The working directory
# Working_directory = "<your_path_here>"

working_directory = "<your_path_here>"
os.chdir(working_directory)


# Df = pd.read_csv("<your_path_here>")
df = pd.read_csv('./alltype_ac_predict_result.csv')
df['GEOID'] = df['GEOID'].apply(lambda x: f"{int(x):011d}")
df['GEOID']= df['GEOID'].astype(str).str[:5]
df['STATENM'] = df['STATE']

df['STATE'] = df['GEOID'].astype(str).str[:2]
df['New_AIRCONDITIONINGrecode'] =df['AIRCONDITIONINGrecode'].combine_first(df['finalAIRCONDITIONINGpre'])

DF = df[['STATE','New_AIRCONDITIONINGrecode','LAT', 'LON']]
DF.rename(columns={'New_AIRCONDITIONINGrecode': 'ACtype'}, inplace=True)



# Define The directory names where the shapefiles will be stored
directories = [
    'boundary/nhgis0075_shapefile_tl2000_us_tract_2000',
    'boundary/nhgis0075_shapefile_tl2010_us_tract_2010',
    'boundary/nhgis0075_shapefile_tl2015_us_tract_2015',
    'boundary/nhgis0075_shapefile_tl2020_us_tract_2020']


# Create Directories if they do not exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Assume DF Is already defined and contains the required data
states = DF['STATE'].unique()

# Loop Through each directory to process all shapefiles
for directory in directories:
    allDF = pd.DataFrame()  # Initialize an empty DataFrame for each directory
    path = os.path.join(working_directory, directory)
    shapefiles = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.shp')]


    # Process Each shapefile
    for shapefile in shapefiles:
        print(f"Processing {shapefile}")
        trcatbd = gpd.read_file(shapefile)  # Read the current shapefile

        statefp_column = 'STATEFP10' if 'STATEFP10' in trcatbd.columns else 'STATEFP'
        geo_id_column = 'GEOID10' if 'GEOID10' in trcatbd.columns else 'GEOID'

        for state in states:
            subdf = DF[DF['STATE'] == state]
            gdf = gpd.GeoDataFrame(subdf, geometry=gpd.points_from_xy(subdf.LON, subdf.LAT), crs='EPSG:4326')
            gdf = gdf.to_crs(trcatbd.crs)  # Reproject the GeoDataFrame to match the tract boundaries

            tactbd_sel = trcatbd[trcatbd[statefp_column] == state]

            interdf = gpd.sjoin(gdf, tactbd_sel, how="inner")  # Spatial join

            # Calculate Percentage
            grouped_percent = interdf.groupby(geo_id_column)['ACtype'].value_counts(normalize=True) * 100
            grouped_percent = grouped_percent.rename('Percentage').reset_index()

            # Calculate Count
            grouped_count = interdf.groupby(geo_id_column)['ACtype'].value_counts().rename('Count').reset_index()

            # Merge Percentage and count data
            mergedf = pd.merge(grouped_percent, grouped_count, on=[geo_id_column, 'ACtype'])

            # Append To the cumulative DataFrame
            allDF = pd.concat([allDF, mergedf], ignore_index=True)

    # Save The compiled DataFrame to a CSV file
    year = directory[-4:]  # Extract year from the directory name
    filename = f'summary_{directory.split("_")[-1]}_{year}.csv'  # Adapted to use the correct directory split for filename
    allDF.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

print("All files processed.")



# Tract level

import geopandas as gpd
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

def process_shapefiles(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        path = os.path.join(working_directory, directory)
        shapefiles = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.shp')]

        for shapefile in shapefiles:
            trcatbd = gpd.read_file(shapefile)
            geo_id_column = 'GEOID10' if 'GEOID10' in trcatbd.columns else 'GEOID'
            statefp_column = 'STATEFP10' if 'STATEFP10' in trcatbd.columns else 'STATEFP'

            for state in states:
                subdf = DF[DF['STATE'] == state]
                gdf = gpd.GeoDataFrame(subdf, geometry=gpd.points_from_xy(subdf.LON, subdf.LAT), crs='EPSG:4326')
                gdf = gdf.to_crs(trcatbd.crs)

                tactbd_sel = trcatbd[trcatbd[statefp_column] == state]
                interdf = gpd.sjoin(gdf, tactbd_sel, how="inner")

                grouped_percent = interdf.groupby(geo_id_column)['ACtype'].value_counts(normalize=True) * 100
                grouped_percent = grouped_percent.rename('Percentage').reset_index()
                grouped_count = interdf.groupby(geo_id_column)['ACtype'].value_counts().rename('Count').reset_index()

                mergedf = pd.merge(grouped_percent, grouped_count, on=[geo_id_column, 'ACtype'])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)

        year = directory[-5:-1]    # Extract year from the directory name
        filename = f'summary_{directory.split("_")[-2]}_{year}.csv'
        allDF.to_csv(os.path.join(working_directory, filename), index=False)
        print(f"Data saved to {os.path.join(working_directory, filename)}")
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")

# Set The working directory and prepare directories list
working_directory = "<your_path_here>"
os.chdir(working_directory)

directories = [
    'boundary/nhgis0075_shapefile_tl2000_us_tract_2000/',
    'boundary/nhgis0075_shapefile_tl2010_us_tract_2010/',
    'boundary/nhgis0075_shapefile_tl2015_us_tract_2015/',
    'boundary/nhgis0075_shapefile_tl2020_us_tract_2020/'
]

df = pd.read_csv('./alltype_ac_predict_result.csv')
df['GEOID'] = df['GEOID'].apply(lambda x: f"{int(x):011d}")
df['GEOID'] = df['GEOID'].astype(str).str[:5]
df['STATE'] = df['GEOID'].astype(str).str[:2]
df.loc[df['AIRCONDITIONINGrecode'].isna() | (df['AIRCONDITIONINGrecode'] == 'Yes'), 'New_AIRCONDITIONINGrecode'] = df['finalAIRCONDITIONINGpre']

DF = df[['STATE', 'New_AIRCONDITIONINGrecode', 'LAT', 'LON']]
DF.rename(columns={'New_AIRCONDITIONINGrecode': 'ACtype'}, inplace=True)

states = DF['STATE'].unique()  # Assume DF is loaded earlier

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Use ProcessPoolExecutor To handle the multiprocessing
with ProcessPoolExecutor() as executor:
    executor.map(process_shapefiles, directories)

print("All files processed.")




# Zipcode level

import geopandas as gpd
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

def process_shapefiles(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        path = os.path.join(working_directory, directory)
        shapefiles = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.shp')]

        state_boundaries = gpd.read_file("./boundary/cb_2020_us_state_500k/cb_2020_us_state_500k.shp")


        for shapefile in shapefiles:
            trcatbd = gpd.read_file(shapefile)
            # Geo_id_column = 'GEOID10' if 'GEOID10' in trcatbd.columns else 'GEOID'
            geo_id_column = 'ZCTA' if 'ZCTA' in trcatbd.columns else ('ZCTA5CE10' if 'ZCTA5CE10' in trcatbd.columns else 'ZCTA5CE20')

            # Checking if the projection are matched
            if trcatbd.crs != state_boundaries.crs:
                state_boundaries = state_boundaries.to_crs(trcatbd.crs)

            # Performing A spatial join to assign states to ZCTAs
            zcta_with_states = gpd.sjoin(trcatbd, state_boundaries, how="left")

            for state in states:
                subdf = DF[DF['STATE'] == state]
                gdf = gpd.GeoDataFrame(subdf, geometry=gpd.points_from_xy(subdf.LON, subdf.LAT), crs='EPSG:4326')
                gdf = gdf.to_crs(trcatbd.crs)


                tactbd_sel = zcta_with_states[zcta_with_states['STATEFP'] == state]

                # Rename 'index_right' If it exists in tactbd_sel DataFrame
                if 'index_right' in tactbd_sel.columns:
                    tactbd_sel = tactbd_sel.drop(columns=['index_right'])


                interdf = gpd.sjoin(gdf, tactbd_sel, how="inner")

                grouped_percent = interdf.groupby(geo_id_column)['ACtype'].value_counts(normalize=True) * 100
                grouped_percent = grouped_percent.rename('Percentage').reset_index()
                grouped_count = interdf.groupby(geo_id_column)['ACtype'].value_counts().rename('Count').reset_index()

                mergedf = pd.merge(grouped_percent, grouped_count, on=[geo_id_column, 'ACtype'])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)

        year = directory[-5:-1]  # Extract year from the directory name
        filename = f'summary_{directory.split("_")[-2]}_{year}.csv'
        allDF.to_csv(os.path.join(working_directory, filename), index=False)
        print(f"Data saved to {os.path.join(working_directory, filename)}")
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")


# Set The working directory and prepare directories list
working_directory = "<your_path_here>"
os.chdir(working_directory)

directories = [
    'boundary/nhgis0077_shapefile_tl2000_us_zcta_2000/',
    'boundary/nhgis0077_shapefile_tl2010_us_zcta_2010/',
    'boundary/nhgis0077_shapefile_tl2015_us_zcta_2015/',
    'boundary/nhgis0077_shapefile_tl2020_us_zcta_2020/'
]


df = pd.read_csv('./alltype_ac_predict_result.csv')
df['GEOID'] = df['GEOID'].apply(lambda x: f"{int(x):011d}")
df['GEOID'] = df['GEOID'].astype(str).str[:5]
df['STATE'] = df['GEOID'].astype(str).str[:2]
df.loc[df['AIRCONDITIONINGrecode'].isna() | (df['AIRCONDITIONINGrecode'] == 'Yes'), 'New_AIRCONDITIONINGrecode'] = df['finalAIRCONDITIONINGpre']


DF = df[['STATE', 'New_AIRCONDITIONINGrecode', 'LAT', 'LON']]
DF.rename(columns={'New_AIRCONDITIONINGrecode': 'ACtype'}, inplace=True)

states = DF['STATE'].unique()  # Assume DF is loaded earlier

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Use ProcessPoolExecutor To handle the multiprocessing
with ProcessPoolExecutor() as executor:
    executor.map(process_shapefiles, directories)

print("All files processed.")



# This is for city level
import geopandas as gpd
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

def process_shapefiles(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        path = os.path.join(working_directory, directory)
        shapefiles = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.shp')]

        state_boundaries = gpd.read_file("./boundary/cb_2020_us_state_500k/cb_2020_us_state_500k.shp")


        for shapefile in shapefiles:
            trcatbd = gpd.read_file(shapefile)
            # Geo_id_column = 'GEOID10' if 'GEOID10' in trcatbd.columns else 'GEOID'
            geo_id_column = 'GEOID10' if 'GEOID10' in trcatbd.columns else 'GEOID'


            # Checking if the projection are matched
            if trcatbd.crs != state_boundaries.crs:
                state_boundaries = state_boundaries.to_crs(trcatbd.crs)
                state_boundaries.rename(columns={'GEOID': 'State'}, inplace=True)

            # Performing A spatial join to assign states to ZCTAs
            zcta_with_states = gpd.sjoin(trcatbd, state_boundaries, how="left")

            for state in states:
                subdf = DF[DF['STATE'] == state]
                gdf = gpd.GeoDataFrame(subdf, geometry=gpd.points_from_xy(subdf.LON, subdf.LAT), crs='EPSG:4326')
                gdf = gdf.to_crs(trcatbd.crs)


                tactbd_sel = zcta_with_states[zcta_with_states['STATEFP'] == state]

                # Rename 'index_right' If it exists in tactbd_sel DataFrame
                if 'index_right' in tactbd_sel.columns:
                    tactbd_sel = tactbd_sel.drop(columns=['index_right'])


                interdf = gpd.sjoin(gdf, tactbd_sel, how="inner")

                grouped_percent = interdf.groupby(['STATEFP', geo_id_column])['ACtype'].value_counts(normalize=True) * 100
                grouped_percent = grouped_percent.rename('Percentage').reset_index()
                grouped_count = interdf.groupby(['STATEFP', geo_id_column])['ACtype'].value_counts().rename('Count').reset_index()

                mergedf = pd.merge(grouped_percent, grouped_count, on=['STATEFP', geo_id_column, 'ACtype'])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)

        year = directory.split("_")[-3][-4:]  # Extract year from the directory name
        filename = f'summary_{directory.split("_")[-1]}_{year}.csv'
        allDF.to_csv(os.path.join(working_directory, filename), index=False)
        print(f"Data saved to {os.path.join(working_directory, filename)}")
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")



# Set The working directory and prepare directories list
working_directory = "<your_path_here>"
os.chdir(working_directory)

directories = [
    # 'boundary/nhgis0077_shapefile_tl2000_us_zcta_2000/',
    'boundary/2015_us_cbsa/',
    'boundary/2010_us_cbsa/',
    'boundary/2020_us_cbsa/'
]



df = pd.read_csv('./alltype_ac_predict_result.csv')
df['GEOID'] = df['GEOID'].apply(lambda x: f"{int(x):011d}")
df['GEOID'] = df['GEOID'].astype(str).str[:5]
df['STATE'] = df['GEOID'].astype(str).str[:2]
df.loc[df['AIRCONDITIONINGrecode'].isna() | (df['AIRCONDITIONINGrecode'] == 'Yes'), 'New_AIRCONDITIONINGrecode'] = df['finalAIRCONDITIONINGpre']


DF = df[['STATE', 'New_AIRCONDITIONINGrecode', 'LAT', 'LON']]
DF.rename(columns={'New_AIRCONDITIONINGrecode': 'ACtype'}, inplace=True)

states = DF['STATE'].unique()  # Assume DF is loaded earlier

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Use ProcessPoolExecutor To handle the multiprocessing
with ProcessPoolExecutor() as executor:
    executor.map(process_shapefiles, directories)

print("All files processed.")




# Adding state and county info to Zip data

import geopandas as gpd
import os

# Define Paths
working_directory = "<your_path_here>"

county_paths = {
    2000: 'Data/boundary/cb_2000_us_county_500k/co99_d00.shp',
    2010: 'Data/boundary/cb_2010_us_county_500k/gz_2010_us_050_00_500k.shp',
    2015: 'Data/boundary/cb_2015_us_county_500k/cb_2015_us_county_500k.shp',
    2020: 'Data/boundary/cb_2020_us_county_500k/cb_2020_us_county_500k.shp'
}

zcta_paths = {
    2000: 'Data/boundary/nhgis0077_shapefile_tl2000_us_zcta_2000/US_zcta_2000.shp',
    2010: 'Data/boundary/nhgis0077_shapefile_tl2010_us_zcta_2010/US_zcta_2010.shp',
    2015: 'Data/boundary/nhgis0077_shapefile_tl2015_us_zcta_2015/US_zcta_2015.shp',
    2020: 'Data/boundary/nhgis0077_shapefile_tl2020_us_zcta_2020/US_zcta_2020.shp'
}

# Output Directory

output_dir = os.path.join(working_directory, 'Data/results/intersections_csv/')
os.makedirs(output_dir, exist_ok=True)

for year in county_paths:
    county_fp = os.path.join(working_directory, county_paths[year])
    zcta_fp = os.path.join(working_directory, zcta_paths[year])

    if not (os.path.exists(county_fp) and os.path.exists(zcta_fp)):
        print(f"Missing files for {year}")
        continue

    print(f"\nProcessing {year}...")

    counties = gpd.read_file(county_fp)
    zctas = gpd.read_file(zcta_fp)

    # Manually Set CRS if counties don't have one
    if counties.crs is None:
        print(f"County file for {year} has no CRS. Assigning EPSG:4269 (NAD83).")
        counties.set_crs(epsg=4269, inplace=True)

    # Reproject ZCTAs To match county CRS
    if zctas.crs != counties.crs:
        zctas = zctas.to_crs(counties.crs)


    # Match CRS
    if counties.crs != zctas.crs:
        zctas = zctas.to_crs(counties.crs)

    # Spatial Intersection
    intersected = gpd.overlay(zctas, counties, how='intersection')

    # Determine ZCTA Column name
    if 'ZCTA' in zctas.columns:
        zcta_col = 'ZCTA'
    elif 'ZCTA5CE10' in zctas.columns:
        zcta_col = 'ZCTA5CE10'
    else:
        zcta_col = 'ZCTA5CE20'

    # Determine County GEOID column
    if 'GEOID' in counties.columns:
        county_col = 'GEOID'
        intersected['GEOID'] = intersected[county_col]
    elif 'GEO_ID' in counties.columns:
        intersected['GEOID'] = intersected['GEO_ID'].str[-5:]
    elif 'STATE' in counties.columns and 'COUNTY' in counties.columns:
        intersected['GEOID'] = intersected['STATE'].astype(str).str.zfill(2) + intersected['COUNTY'].astype(str).str.zfill(3)
    elif 'STATEFP' in counties.columns and 'COUNTYFP' in counties.columns:
        intersected['GEOID'] = intersected['STATEFP'] + intersected['COUNTYFP']
    else:
        raise ValueError(f"Cannot find GEOID columns for year {year}")

    # Get ZCTA Code
    intersected['ZCTA'] = intersected[zcta_col].astype(str)

    # Keep Only relevant columns
    final_df = intersected[['GEOID', 'ZCTA']].drop_duplicates()

    # Save To CSV
    out_fp = os.path.join(output_dir, f'key_county_zip_{year}.csv')
    final_df.to_csv(out_fp, index=False)
    print(f"Saved {out_fp}")



import pandas as pd
import os

# Directories
intersect_dir = "<your_path_here>"
summary_dir = "<your_path_here>"
output_dir = "<your_path_here>"


os.makedirs(output_dir, exist_ok=True)

years = [2000, 2010, 2015, 2020]

for year in years:
    try:
        # File Paths
        intersect_fp = os.path.join(intersect_dir, f'key_county_zip_{year}.csv')
        summary_fp = os.path.join(summary_dir, f'summary_zcta_{year}.csv')

        # Load Data
        df_intersect = pd.read_csv(intersect_fp, dtype=str)
        df_summary = pd.read_csv(summary_fp, dtype=str)

        # Identify ZCTA Column in summary file
        zcta_col = None
        for candidate in ['ZCTA', 'ZCTA5CE10', 'ZCTA5CE20']:
            if candidate in df_summary.columns:
                zcta_col = candidate
                break

        if zcta_col is None:
            raise ValueError(f"No valid ZCTA column found in summary file for {year}")

        # Rename To "ZCTA" for consistency
        df_summary = df_summary.rename(columns={zcta_col: 'ZCTA'})

        # Merge
        df_merged = df_summary.merge(df_intersect, on='ZCTA', how='left')

        # Round 'Percentage' To 2 decimal places
        if 'Percentage' in df_merged.columns:
            df_merged['Percentage'] = pd.to_numeric(df_merged['Percentage'], errors='coerce').round(2)

        # Ensure Final column order
        expected_columns = ['GEOID', 'ZCTA', 'ACtype', 'Percentage', 'Count']
        for col in expected_columns:
            if col not in df_merged.columns:
                df_merged[col] = pd.NA  # Fill with NA if missing

        df_merged = df_merged[expected_columns]
        df_merged.GEOID = df_merged.GEOID.apply(lambda x: '{0:0>5}'.format(x))
        df_merged.GEOID = df_merged.GEOID.astype(str)
        df_merged.ZCTA= df_merged.ZCTA.astype(str)

        # Save
        output_fp = os.path.join(output_dir, f'summary_zcta_with_county_{year}.csv')
        df_merged.to_csv(output_fp, index=False)
        print(f"Saved: {output_fp}")


    except Exception as e:
        print(f"Error processing year {year}: {e}")



import pandas as pd
import os

# Input And output paths
input_files = {
    2010: "<your_path_here>",
    2015: "<your_path_here>",
    2020: "<your_path_here>",
}

output_dir = "<your_path_here>"
os.makedirs(output_dir, exist_ok=True)

# Desired Column order
final_columns = ['GEOIDCN', 'GEOID', 'ACtype', 'Percentage', 'Count']

for year, input_path in input_files.items():
    # Load File
    df = pd.read_csv(input_path, dtype={"GEOID": str})

    # Filter For valid 11-digit GEOIDs
    df = df[df["GEOID"].str.len() == 11]

    # Create GEOIDCN Column
    df["GEOIDCN"] = df["GEOID"].str[:5]

    # Round Percentage To 2 decimal places
    if "Percentage" in df.columns:
        df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce").round(2)

    # Ensure All expected columns exist (fill missing ones with NA)
    for col in final_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder Columns
    df = df[final_columns]

    # Save Output
    output_path = os.path.join(output_dir, f"summary_tract_{year}.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved ordered file to: {output_path}")



# Intersect with tract and city files
import geopandas as gpd
import os
from pathlib import Path

# Input Paths
cbsa_paths = {
    2000: "<your_path_here>",
    2010: "<your_path_here>",
    2015: "<your_path_here>",
    2020: "<your_path_here>",
}

tract_paths = {
    2000: "<your_path_here>",
    2010: "<your_path_here>",
    2015: "<your_path_here>",
    2020: "<your_path_here>",
}

import os
import geopandas as gpd
from pathlib import Path

output_dir = "<your_path_here>"
os.makedirs(output_dir, exist_ok=True)

for year in [2020]:#2000, 2010, 2015,
    cbsa_dir = cbsa_paths[year]
    tract_dir = tract_paths[year]

    cbsa_shp = next(Path(cbsa_dir).glob("*.shp"), None)
    tract_shp = next(Path(tract_dir).glob("*.shp"), None)

    if cbsa_shp is None or tract_shp is None:
        print(f"Missing shapefile for year {year}")
        continue

    cbsa = gpd.read_file(cbsa_shp)
    tract = gpd.read_file(tract_shp)

    # --- CBSA Standardization ---
    if year == 2000:
        if 'MSACMSA' not in cbsa.columns:
            print(f"Skipping year {year}: missing MSACMSA")
            continue
        cbsa['CBSAFP'] = cbsa['MSACMSA'].astype(str)
        cbsa['NAME'] = 'Unknown'  # Placeholder
        cbsa = cbsa[['CBSAFP', 'NAME', 'geometry']]
    elif year == 2010:
        if 'CBSAFP10' not in cbsa.columns or 'NAME10' not in cbsa.columns:
            print(f"Skipping year {year}: missing CBSAFP10 or NAME10")
            continue
        cbsa = cbsa.rename(columns={'CBSAFP10': 'CBSAFP', 'NAME10': 'NAME'})
        cbsa = cbsa[['CBSAFP', 'NAME', 'geometry']]
    else:
        cbsa = cbsa.rename(columns={'GEOID': 'GEOIDCN'})
        cbsa = cbsa[['CBSAFP', 'NAME', 'geometry']]



    # --- Tract Standardization ---
    if year == 2000:
        tract_id_col = 'GISJOIN' if 'GISJOIN' in tract.columns else None
    else:
        tract_id_col = 'GEOID' if 'GEOID' in tract.columns else 'GEOID10' if 'GEOID10' in tract.columns else None

    if not tract_id_col:
        print(f"Skipping year {year}: missing GEOID, GEOID10, or GISJOIN")
        continue

    tract['GEOID'] = tract[tract_id_col].astype(str).str.zfill(11)
    tract = tract[tract['GEOID'].str.len() == 11]
    tract['GEOIDCN'] = tract['GEOID'].str[:5]
    tract = tract[['GEOID', 'GEOIDCN', 'geometry']]

    # Match CRS
    if cbsa.crs != tract.crs:
        tract = tract.to_crs(cbsa.crs)

    # Intersect
    intersect = gpd.overlay(tract, cbsa, how='intersection')

    # Save
    output_fp = os.path.join(output_dir, f"key_tract_city_{year}.csv")
    intersect[['GEOIDCN', 'GEOID', 'CBSAFP', 'NAME']].to_csv(output_fp, index=False)
    print(f"Saved: {output_fp}")



# Matching the  tract level AC with the city code

tractlevel = ["<your_path_here>",
              "<your_path_here>",
              "<your_path_here>"]

metroindex = ["<your_path_here>",
              "<your_path_here>",
              "<your_path_here>"]

# Output Directory
output_dir = "<your_path_here>"

# Merge And save
for tfile, mfile in zip(tractlevel, metroindex):
    year = tfile.split('_')[-1].split('.')[0]  # Extract year from filename

    # Read Data
    df_tract = pd.read_csv(tfile)
    df_metro = pd.read_csv(mfile)

    # Merge By 'GEOID'
    df_merged = pd.merge(df_tract, df_metro, on=['GEOID'], how='inner')

    # Output Path
    output_path = os.path.join(output_dir, f"summary_metro_{year}.csv")

    # Save
    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged file for {year} to: {output_path}")



