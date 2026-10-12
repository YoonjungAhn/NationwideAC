#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:47:28 2025

@author: y943a214

Step 5 of the pipeline: aggregate the property-level AC predictions to
census geographies, and build the keys used for the final datasets in Data/.

Steps (turn each on/off in the RUN_* settings below)
-----
1. Count properties by AC type within each census tract, ZIP Code Tabulation
   Area (ZCTA), metro area (CBSA) and block group, for the 2000, 2010, 2015
   and 2020 boundaries                          -> RESULTS_DIR/summary_<level>_<year>.csv
2. Key between ZCTAs and counties              -> RESULTS_DIR/intersections_csv/key_county_zip_<year>.csv
3. Add county GEOID to the ZCTA summaries      -> FINAL_DIR/summary_zcta_with_county_<year>.csv
4. Final column order for the tract summaries  -> FINAL_DIR/summary_tract_<year>.csv
5. Key between tracts and metro areas          -> RESULTS_DIR/intersections_csv/key_tract_city_<year>.csv
   (then run UShouse.Rmd to adjust the counts to Census housing units)
6. Add metro area to the adjusted tract data   -> RESULTS_DIR/summary_metro_adjusted<year>.csv
   (needs FINAL_DIR/summary_tract_adjusted_<year>.csv from UShouse.Rmd)

Input files
-----------
1. alltype_ac_predict_result.csv : property-level AC predictions, the output of
   Alltype_ACesimtation_HPC_predict.py (in WORKING_DIR).
2. Boundary shapefiles (.shp) in WORKING_DIR/boundary/ for census tracts,
   ZCTAs and block groups, 2000/2010/2015/2020. These were downloaded from
   NHGIS (IPUMS National Historical Geographic Information System,
   https://www.nhgis.org) -> "GIS Files" -> select the geographic level and
   year. NHGIS names every download "nhgisXXXX_..." with your own extract
   number (ours were nhgis0075 for tracts, nhgis0077 for ZCTAs and nhgis0080
   for block groups), so rename the folders below to match your extract.
3. CBSA (metro/micro area) shapefiles for 2000, 2010, 2015, 2020 and county
   and state cartographic boundary files (cb_*_us_county_500k.shp,
   cb_2020_us_state_500k.shp) from the U.S. Census Bureau
   (https://www.census.gov/geographies/mapping-files.html).

Step 1 uses several processes (ProcessPoolExecutor) and was run on Linux (HPC).
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# =============================================================================
# 0. Settings
# =============================================================================
WORKING_DIR = '<your_path_here>'                    # holds alltype_ac_predict_result.csv and boundary/
RESULTS_DIR = '<your_path_here>'                    # summary files are written here
FINAL_DIR = os.path.join(RESULTS_DIR, 'FINALdata')  # final, formatted files
KEY_DIR = os.path.join(RESULTS_DIR, 'intersections_csv')

# Which steps to run
RUN_STEP1_SUMMARIES = True
RUN_STEP2_ZCTA_COUNTY_KEY = True
RUN_STEP3_ZCTA_WITH_COUNTY = True
RUN_STEP4_TRACT_FORMAT = True
RUN_STEP5_TRACT_CITY_KEY = True
RUN_STEP6_METRO_SUMMARY = False   # run after UShouse.Rmd

# Boundary folders for step 1 (relative to WORKING_DIR), one per boundary year.
# Replace "nhgis0075"/"nhgis0077"/"nhgis0080" with your own NHGIS extract numbers.
TRACT_DIRS = [
    'boundary/nhgis0075_shapefile_tl2000_us_tract_2000/',
    'boundary/nhgis0075_shapefile_tl2010_us_tract_2010/',
    'boundary/nhgis0075_shapefile_tl2015_us_tract_2015/',
    'boundary/nhgis0075_shapefile_tl2020_us_tract_2020/',
]
ZCTA_DIRS = [
    'boundary/nhgis0077_shapefile_tl2000_us_zcta_2000/',
    'boundary/nhgis0077_shapefile_tl2010_us_zcta_2010/',
    'boundary/nhgis0077_shapefile_tl2015_us_zcta_2015/',
    'boundary/nhgis0077_shapefile_tl2020_us_zcta_2020/',
]
CBSA_DIRS = [            # Census CBSA (metro/micro area) shapefiles
    'boundary/2010_us_cbsa/',
    'boundary/2015_us_cbsa/',
    'boundary/2020_us_cbsa/',
]
BLOCKGROUP_DIRS = [
    'boundary/nhgis0080_shapefile_tl2000_us_blck_grp_2000/',
    'boundary/nhgis0080_shapefile_tl2010_us_blck_grp_2010/',
    'boundary/nhgis0080_shapefile_tl2015_us_blck_grp_2015/',
    'boundary/nhgis0080_shapefile_tl2020_us_blck_grp_2020/',
]
STATE_SHP = 'boundary/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'

# Boundary files for steps 2 and 5 (relative to WORKING_DIR)
COUNTY_SHP = {
    2000: 'boundary/cb_2000_us_county_500k/co99_d00.shp',
    2010: 'boundary/cb_2010_us_county_500k/gz_2010_us_050_00_500k.shp',
    2015: 'boundary/cb_2015_us_county_500k/cb_2015_us_county_500k.shp',
    2020: 'boundary/cb_2020_us_county_500k/cb_2020_us_county_500k.shp',
}
ZCTA_SHP = {
    2000: 'boundary/nhgis0077_shapefile_tl2000_us_zcta_2000/US_zcta_2000.shp',
    2010: 'boundary/nhgis0077_shapefile_tl2010_us_zcta_2010/US_zcta_2010.shp',
    2015: 'boundary/nhgis0077_shapefile_tl2015_us_zcta_2015/US_zcta_2015.shp',
    2020: 'boundary/nhgis0077_shapefile_tl2020_us_zcta_2020/US_zcta_2020.shp',
}
CBSA_FOLDER = {
    2000: 'boundary/2000_us_msa_cmsa',
    2010: 'boundary/2010_us_cbsa',
    2015: 'boundary/2015_us_cbsa',
    2020: 'boundary/2020_us_cbsa',
}
TRACT_FOLDER = {year: f'boundary/nhgis0075_shapefile_tl{year}_us_tract_{year}' for year in [2000, 2010, 2015, 2020]}


# =============================================================================
# Helper functions
# =============================================================================
def load_points():
    """Property points with their state FIPS and final AC type.

    ACtype is the recorded AC type; where it is missing (or only "Yes"),
    the predicted type from the model (finalAIRCONDITIONINGpre) is used.
    """
    df = pd.read_csv(os.path.join(WORKING_DIR, 'alltype_ac_predict_result.csv'))
    df['GEOID'] = df['GEOID'].apply(lambda x: f"{int(x):011d}")
    df['STATE'] = df['GEOID'].str[:2]   # 2-digit state FIPS

    df['ACtype'] = np.where(
        df['AIRCONDITIONINGrecode'].isna() | (df['AIRCONDITIONINGrecode'] == 'Yes'),
        df['finalAIRCONDITIONINGpre'],
        df['AIRCONDITIONINGrecode'])
    return df[['PROPID', 'STATE', 'ACtype', 'LAT', 'LON']]


def year_from_dir(directory):
    """Boundary year: the last 4-digit number in the folder name."""
    return re.findall(r'\d{4}', directory)[-1]


def list_shapefiles(directory):
    path = os.path.join(WORKING_DIR, directory)
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.shp')]


def format_fips(series, width):
    """Zero-pad FIPS codes to `width` digits."""
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().all():
        return numeric.astype('Int64').astype(str).str.zfill(width)
    return series.astype(str).str.zfill(width)


def prepare_tract_boundary_columns(trcatbd, shapefile):
    """Return state and tract id columns, creating them for older NHGIS files."""
    columns = set(trcatbd.columns)

    state_candidates = ['STATEFP10', 'STATEFP00', 'STATEFP', 'STATEA', 'STATE']
    geo_id_candidates = ['GEOID10', 'GEOID00', 'GEOID', 'STFID', 'GISJOIN']

    statefp_column = next((col for col in state_candidates if col in columns), None)
    geo_id_column = next((col for col in geo_id_candidates if col in columns), None)

    # Build an 11-digit tract GEOID from state + county + tract codes if needed
    if geo_id_column is None:
        tract_parts = [
            ('STATEA', 'COUNTYA', 'TRACTA'),
            ('STATEFP10', 'COUNTYFP10', 'TRACTCE10'),
            ('STATEFP00', 'COUNTYFP00', 'TRACTCE00'),
            ('STATEFP', 'COUNTYFP', 'TRACTCE'),
            ('STATE', 'COUNTY', 'TRACT'),
        ]
        for state_col, county_col, tract_col in tract_parts:
            if {state_col, county_col, tract_col}.issubset(columns):
                trcatbd['GEOID'] = (format_fips(trcatbd[state_col], 2)
                                    + format_fips(trcatbd[county_col], 3)
                                    + format_fips(trcatbd[tract_col], 6))
                geo_id_column = 'GEOID'
                break

    if geo_id_column is None:
        raise ValueError(f"Could not find or build a tract id column for {shapefile}. "
                         f"Available columns: {list(trcatbd.columns)}")

    if statefp_column is None:
        if geo_id_column == 'GISJOIN':
            trcatbd['STATEFP'] = trcatbd['GISJOIN'].astype(str).str[1:3]
        else:
            trcatbd['STATEFP'] = trcatbd[geo_id_column].astype(str).str[:2]
        statefp_column = 'STATEFP'

    trcatbd[statefp_column] = format_fips(trcatbd[statefp_column], 2)
    return trcatbd, statefp_column, geo_id_column


def count_by_polygon(points, polygons, group_cols):
    """Spatial join of property points to polygons, then the count and
    percentage of each AC type within each polygon."""
    gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.LON, points.LAT), crs='EPSG:4326')
    gdf = gdf.to_crs(polygons.crs)   # match the boundary projection
    interdf = gpd.sjoin(gdf, polygons, how="inner")

    grouped_percent = interdf.groupby(group_cols)['ACtype'].value_counts(normalize=True) * 100
    grouped_percent = grouped_percent.rename('Percentage').reset_index()
    grouped_count = interdf.groupby(group_cols)['ACtype'].value_counts().rename('Count').reset_index()
    return pd.merge(grouped_percent, grouped_count, on=group_cols + ['ACtype'])


def add_state_to_polygons(polygons):
    """ZCTAs and CBSAs have no state code: assign one by spatial join with state boundaries."""
    state_boundaries = gpd.read_file(os.path.join(WORKING_DIR, STATE_SHP))
    if polygons.crs != state_boundaries.crs:
        state_boundaries = state_boundaries.to_crs(polygons.crs)
    joined = gpd.sjoin(polygons, state_boundaries, how="left")
    return joined.drop(columns=['index_right'])   # so it can be joined again


def save_summary(allDF, level, directory):
    filename = os.path.join(RESULTS_DIR, f'summary_{level}_{year_from_dir(directory)}.csv')
    allDF.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# -----------------------------------------------------------------------------
# Step 1 functions: one boundary year (folder) per call, run in parallel.
# They use the global POINTS and STATES set in the main block.
# -----------------------------------------------------------------------------
def summarize_tracts(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        for shapefile in list_shapefiles(directory):
            trcatbd = gpd.read_file(shapefile)
            trcatbd, statefp_column, geo_id_column = prepare_tract_boundary_columns(trcatbd, shapefile)

            # One state at a time to keep the spatial join small
            for state in STATES:
                tracts_state = trcatbd[trcatbd[statefp_column] == state]
                if tracts_state.empty:
                    continue
                mergedf = count_by_polygon(POINTS[POINTS['STATE'] == state], tracts_state, [geo_id_column])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)
        save_summary(allDF, 'tract', directory)
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")


def summarize_zctas(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        for shapefile in list_shapefiles(directory):
            zcta = gpd.read_file(shapefile)
            # ZCTA id column differs by year
            geo_id_column = 'ZCTA' if 'ZCTA' in zcta.columns else ('ZCTA5CE10' if 'ZCTA5CE10' in zcta.columns else 'ZCTA5CE20')
            zcta_with_states = add_state_to_polygons(zcta)

            for state in STATES:
                zcta_state = zcta_with_states[zcta_with_states['STATEFP'] == state]
                mergedf = count_by_polygon(POINTS[POINTS['STATE'] == state], zcta_state, [geo_id_column])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)
        save_summary(allDF, 'zcta', directory)
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")


def summarize_cbsas(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        for shapefile in list_shapefiles(directory):
            cbsa = gpd.read_file(shapefile)
            geo_id_column = 'GEOID10' if 'GEOID10' in cbsa.columns else 'GEOID'
            cbsa_with_states = add_state_to_polygons(cbsa)

            for state in STATES:
                cbsa_state = cbsa_with_states[cbsa_with_states['STATEFP'] == state]
                mergedf = count_by_polygon(POINTS[POINTS['STATE'] == state], cbsa_state, ['STATEFP', geo_id_column])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)
        save_summary(allDF, 'cbsa', directory)
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")


def summarize_blockgroups(directory):
    try:
        print(f"Processing directory: {directory}")
        allDF = pd.DataFrame()
        for shapefile in list_shapefiles(directory):
            blkgrp = gpd.read_file(shapefile)

            # Block group id: STFID (2000) or GEOID10 / GEOID (renamed to GEOID10)
            geo_id_column = 'GEOID10' if 'GEOID10' in blkgrp.columns else (
                'STFID' if 'STFID' in blkgrp.columns else 'GEOID')
            if geo_id_column == 'GEOID':
                blkgrp = blkgrp.rename(columns={'GEOID': 'GEOID10'})
                geo_id_column = 'GEOID10'
            if 'STATEFP' in blkgrp.columns:
                blkgrp = blkgrp.rename(columns={'STATEFP': 'STATE'})

            blkgrp_with_states = add_state_to_polygons(blkgrp)

            for state in STATES:
                blkgrp_state = blkgrp_with_states[blkgrp_with_states['STATEFP'] == state]
                mergedf = count_by_polygon(POINTS[POINTS['STATE'] == state], blkgrp_state, [geo_id_column])
                allDF = pd.concat([allDF, mergedf], ignore_index=True)
        save_summary(allDF, 'grp', directory)
    except Exception as e:
        print(f"An error occurred in {directory}: {e}")


# -----------------------------------------------------------------------------
# Steps 2-6
# -----------------------------------------------------------------------------
def make_zcta_county_key(years):
    """Step 2: which county (5-digit GEOID) each ZCTA overlaps."""
    os.makedirs(KEY_DIR, exist_ok=True)
    for year in years:
        county_fp = os.path.join(WORKING_DIR, COUNTY_SHP[year])
        zcta_fp = os.path.join(WORKING_DIR, ZCTA_SHP[year])
        if not (os.path.exists(county_fp) and os.path.exists(zcta_fp)):
            print(f"Missing files for {year}")
            continue
        print(f"\nProcessing {year}...")

        counties = gpd.read_file(county_fp)
        zctas = gpd.read_file(zcta_fp)

        # The 2000 county file has no projection: assign NAD83
        if counties.crs is None:
            print(f"County file for {year} has no CRS. Assigning EPSG:4269 (NAD83).")
            counties.set_crs(epsg=4269, inplace=True)
        if zctas.crs != counties.crs:
            zctas = zctas.to_crs(counties.crs)

        intersected = gpd.overlay(zctas, counties, how='intersection')

        # ZCTA id column differs by year
        if 'ZCTA' in zctas.columns:
            zcta_col = 'ZCTA'
        elif 'ZCTA5CE10' in zctas.columns:
            zcta_col = 'ZCTA5CE10'
        else:
            zcta_col = 'ZCTA5CE20'

        # County id column differs by year: build a 5-digit GEOID
        if 'GEOID' in counties.columns:
            pass  # already a 5-digit county GEOID
        elif 'GEO_ID' in counties.columns:
            intersected['GEOID'] = intersected['GEO_ID'].str[-5:]
        elif 'STATE' in counties.columns and 'COUNTY' in counties.columns:
            intersected['GEOID'] = intersected['STATE'].astype(str).str.zfill(2) + intersected['COUNTY'].astype(str).str.zfill(3)
        elif 'STATEFP' in counties.columns and 'COUNTYFP' in counties.columns:
            intersected['GEOID'] = intersected['STATEFP'] + intersected['COUNTYFP']
        else:
            raise ValueError(f"Cannot find GEOID columns for year {year}")

        intersected['ZCTA'] = intersected[zcta_col].astype(str)
        out_fp = os.path.join(KEY_DIR, f'key_county_zip_{year}.csv')
        intersected[['GEOID', 'ZCTA']].drop_duplicates().to_csv(out_fp, index=False)
        print(f"Saved {out_fp}")


def add_county_to_zcta_summary(years):
    """Step 3: ZCTA summary + county GEOID -> GEOID, ZCTA, ACtype, Percentage, Count."""
    os.makedirs(FINAL_DIR, exist_ok=True)
    expected_columns = ['GEOID', 'ZCTA', 'ACtype', 'Percentage', 'Count']
    for year in years:
        try:
            df_intersect = pd.read_csv(os.path.join(KEY_DIR, f'key_county_zip_{year}.csv'), dtype=str)
            df_summary = pd.read_csv(os.path.join(RESULTS_DIR, f'summary_zcta_{year}.csv'), dtype=str)

            # Rename the ZCTA column to "ZCTA"
            zcta_col = next((c for c in ['ZCTA', 'ZCTA5CE10', 'ZCTA5CE20'] if c in df_summary.columns), None)
            if zcta_col is None:
                raise ValueError(f"No valid ZCTA column found in summary file for {year}")
            df_summary = df_summary.rename(columns={zcta_col: 'ZCTA'})

            df_merged = df_summary.merge(df_intersect, on='ZCTA', how='left')
            df_merged['Percentage'] = pd.to_numeric(df_merged['Percentage'], errors='coerce').round(2)
            for col in expected_columns:
                if col not in df_merged.columns:
                    df_merged[col] = pd.NA
            df_merged = df_merged[expected_columns]
            df_merged['GEOID'] = df_merged['GEOID'].apply(lambda x: '{0:0>5}'.format(x)).astype(str)
            df_merged['ZCTA'] = df_merged['ZCTA'].astype(str)

            output_fp = os.path.join(FINAL_DIR, f'summary_zcta_with_county_{year}.csv')
            df_merged.to_csv(output_fp, index=False)
            print(f"Saved: {output_fp}")
        except Exception as e:
            print(f"Error processing year {year}: {e}")


def format_tract_summary(years):
    """Step 4: tract summary -> GEOIDCN, GEOID, ACtype, Percentage, Count."""
    os.makedirs(FINAL_DIR, exist_ok=True)
    final_columns = ['GEOIDCN', 'GEOID', 'ACtype', 'Percentage', 'Count']
    for year in years:
        df = pd.read_csv(os.path.join(RESULTS_DIR, f'summary_tract_{year}.csv'), dtype={"GEOID": str, "GEOID10": str})
        if "GEOID" in df.columns:
            df["GEOID"] = df["GEOID"].astype(str)
        elif "GEOID10" in df.columns:
            df["GEOID"] = df["GEOID10"].astype(str)
        else:
            raise ValueError("Neither 'GEOID' nor 'GEOID10' column found in dataframe.")

        df["GEOIDCN"] = df["GEOID"].str[:5]   # county FIPS
        df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce").round(2)
        for col in final_columns:
            if col not in df.columns:
                df[col] = pd.NA

        output_path = os.path.join(FINAL_DIR, f"summary_tract_{year}.csv")
        df[final_columns].to_csv(output_path, index=False)
        print(f"Saved ordered file to: {output_path}")


def make_tract_city_key(years):
    """Step 5: which metro area (CBSA) each tract overlaps."""
    os.makedirs(KEY_DIR, exist_ok=True)
    for year in years:
        cbsa_shp = next(Path(WORKING_DIR, CBSA_FOLDER[year]).glob("*.shp"), None)
        tract_shp = next(Path(WORKING_DIR, TRACT_FOLDER[year]).glob("*.shp"), None)
        if cbsa_shp is None or tract_shp is None:
            print(f"Missing shapefile for year {year}")
            continue

        cbsa = gpd.read_file(cbsa_shp)
        tract = gpd.read_file(tract_shp)

        # CBSA id and name columns differ by year: use CBSAFP and NAME
        if year == 2000:
            if 'MSACMSA' not in cbsa.columns:
                print(f"Skipping year {year}: missing MSACMSA")
                continue
            cbsa['CBSAFP'] = cbsa['MSACMSA'].astype(str)
            cbsa['NAME'] = 'Unknown'  # the 2000 file has no names
        elif year == 2010:
            if 'CBSAFP10' not in cbsa.columns or 'NAME10' not in cbsa.columns:
                print(f"Skipping year {year}: missing CBSAFP10 or NAME10")
                continue
            cbsa = cbsa.rename(columns={'CBSAFP10': 'CBSAFP', 'NAME10': 'NAME'})
        cbsa = cbsa[['CBSAFP', 'NAME', 'geometry']]

        # Tract id: GISJOIN (2000) or GEOID / GEOID10
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
        if cbsa.crs != tract.crs:
            tract = tract.to_crs(cbsa.crs)

        intersect = gpd.overlay(tract, cbsa, how='intersection')
        output_fp = os.path.join(KEY_DIR, f"key_tract_city_{year}.csv")
        intersect[['GEOIDCN', 'GEOID', 'CBSAFP', 'NAME']].to_csv(output_fp, index=False)
        print(f"Saved: {output_fp}")


def make_metro_summary(years):
    """Step 6: adjusted tract data (from UShouse.Rmd) + metro area."""
    for year in years:
        df_tract = pd.read_csv(os.path.join(FINAL_DIR, f'summary_tract_adjusted_{year}.csv'), dtype={"GEOID": str, "GEOID10": str})
        df_metro = pd.read_csv(os.path.join(KEY_DIR, f'key_tract_city_{year}.csv'), dtype={"GEOID": str, "GEOID10": str})
        if "GEOID" in df_tract.columns:
            df_tract["GEOID"] = df_tract["GEOID"].astype(str)
        elif "GEOID10" in df_tract.columns:
            df_tract["GEOID"] = df_tract["GEOID10"].astype(str)
        else:
            raise ValueError("Neither 'GEOID' nor 'GEOID10' column found in dataframe.")

        df_merged = pd.merge(df_tract, df_metro, on=['GEOID', 'GEOIDCN'], how='inner')
        df_merged = df_merged[['GEOID', 'GEOIDCN', 'CBSAFP', 'NAME', 'ACtype', 'Adjusted_Count', 'Adjusted_Percentage']]

        output_path = os.path.join(RESULTS_DIR, f"summary_metro_adjusted{year}.csv")
        df_merged.to_csv(output_path, index=False)
        print(f"Saved merged file for {year} to: {output_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    os.chdir(WORKING_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: count properties by AC type in each geography (one process per boundary year)
    if RUN_STEP1_SUMMARIES:
        POINTS = load_points()
        STATES = POINTS['STATE'].unique()
        for summarize, dirs in [(summarize_tracts, TRACT_DIRS),
                                (summarize_zctas, ZCTA_DIRS),
                                (summarize_cbsas, CBSA_DIRS),
                                (summarize_blockgroups, BLOCKGROUP_DIRS)]:
            with ProcessPoolExecutor() as executor:
                list(executor.map(summarize, dirs))
        print("All files processed.")

    if RUN_STEP2_ZCTA_COUNTY_KEY:
        make_zcta_county_key([2000, 2010, 2015, 2020])
    if RUN_STEP3_ZCTA_WITH_COUNTY:
        add_county_to_zcta_summary([2000, 2010, 2015, 2020])
    if RUN_STEP4_TRACT_FORMAT:
        format_tract_summary([2010, 2015, 2020])
    if RUN_STEP5_TRACT_CITY_KEY:
        make_tract_city_key([2010, 2015, 2020])
    if RUN_STEP6_METRO_SUMMARY:
        make_metro_summary([2010, 2015, 2020])
