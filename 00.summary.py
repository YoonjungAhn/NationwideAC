# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:43:50 2022

@author: yoah2447
"""
#building density : https://towardsdatascience.com/calculating-building-density-in-r-with-osm-data-e9d85c701e19
#land surface temperature: https://github.com/pylandtemp/pylandtemp/blob/master/tutorials/Tutorial_1-%20load_landsat8_image_from_amazon_and_google_cloud.ipynb

import pandas as pd
import geopandas
import glob

#making data summary

#files = [i for i in glob.glob('C:/Yoonjung/Data/ztrax_2021_extraction_county_areatypes/ztrax_2021_extraction_county_areatypes_'+'*.csv') ]

#DF =pd.DataFrame()
for f in files:
    df = pd.read_csv(f)
    try:
        maxdf = df.groupby(['RowID','BuildingOrImprovementNumber'])['BuildingAreaStndCode','BuildingAreaSqFt'].max().reset_index()
        ztraxdf=df.merge(maxdf,on=['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'],how='right').drop_duplicates(subset = ['RowID','BuildingOrImprovementNumber','BuildingAreaSqFt'])
        ztraxdf['numofbuild'] = 1
        
        ACDF = ztraxdf.groupby(["FIPS",'AirConditioningTypeorSystemStndCode'],dropna=False)['numofbuild'].sum().reset_index()
        ACDF['ACpercent']  =( ACDF['numofbuild'] / ACDF.groupby('FIPS')['numofbuild'].transform('sum'))*100
        DF = pd.concat([DF,ACDF], axis = 0 )
    except KeyError:
        print(f)
    
#DF.to_csv('C:/Yoonjung/research/03_nationwideAC\DATA/ACtypessummary.csv')

DF = pd.read_csv('C:/Yoonjung/research/03_nationwideAC\DATA/ACtypessummary.csv')
DF['FIPS'] = DF['FIPS'].astype(int).astype(str)
DF['FIPS'] =DF['FIPS'].apply(lambda x: '{0:0>5}'.format(x))

#mapping for missing 
import matplotlib.pyplot as plt
usa = gpd.read_file('C:/Yoonjung/research/03_nationwideAC/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
usa.tail() #last 5 records in dataframe
usa['FIPS'] = usa.STATEFP + usa.COUNTYFP
usa.FIPS = usa.FIPS.astype(int).astype(str)

missingdf = DF.loc[DF.AirConditioningTypeorSystemStndCode.isna()]

missingus = usa.merge(missingdf, on =['FIPS'], how='left')

# Initialize the figure

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

title = 'Air Conditioning Ownership Data Missing Rate'
col = 'case_growth_rate'
source = 'Source: ZTRAX \nGrowth Rate = not indicated / total number of properties'
vmin = missingus['ACpercent'].min()
vmax = vmax=missingus['ACpercent'].max()
cmap = 'copper_r'

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 8))

# Remove the axis
ax.axis('off')

missingus.plot(column='ACpercent', ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap, missing_kwds={'color': 'lightgrey'})

# Add a title
ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
ax.set_xlim([-125,-65])
ax.set_ylim([25,50])
# Create an annotation for the data source
ax.annotate(source, xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', 
            verticalalignment='bottom', fontsize=10)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

# Empty array for the data range
sm._A = []

# Add the colorbar to the figure
cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)


#mapping for prevalence
DF['ACprevalence'] = DF.apply(lambda x: "noAC" if x.AirConditioningTypeorSystemStndCode=='VN' else ( "noAC"  if x.AirConditioningTypeorSystemStndCode =="NO" else "yesAC"), axis = 1)
ACprevalence = DF.loc[(DF.AirConditioningTypeorSystemStndCode.notna())&(DF.ACprevalence=='yesAC')].groupby(["FIPS","ACprevalence"])["ACpercent"].sum().reset_index()

prevalenceus = usa.merge(ACprevalence, on =['FIPS'], how='left')
# Initialize the figure

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

title = 'Air Conditioning Prevalence Rate'
col = 'case_growth_rate'
source = 'Source: ZTRAX \nGrowth Rate = any AC types / total number of properties'
vmin = prevalenceus['ACpercent'].min()
vmax = prevalenceus['ACpercent'].max()
cmap = 'Blues'

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 8))

# Remove the axis
ax.axis('off')

prevalenceus.plot(column='ACpercent', ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap, missing_kwds={'color': 'lightgrey'})

# Add a title
ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
ax.set_xlim([-125,-65])
ax.set_ylim([25,50])
# Create an annotation for the data source
ax.annotate(source, xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', 
            verticalalignment='bottom', fontsize=10)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

# Empty array for the data range
sm._A = []

# Add the colorbar to the figure
cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)


#mapping for prevalence
DF['ACprevalence'] = DF.apply(lambda x: "noAC" if x.AirConditioningTypeorSystemStndCode=='VN' else ( "noAC"  if x.AirConditioningTypeorSystemStndCode =="NO" else "yesAC"), axis = 1)
ACprevalence = DF.loc[(DF.AirConditioningTypeorSystemStndCode.notna())&(DF.ACprevalence=='noAC')].groupby(["FIPS","ACprevalence"])["ACpercent"].sum().reset_index()

prevalenceus = usa.merge(ACprevalence, on =['FIPS'], how='left')
# Initialize the figure

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

title = 'Lower Air Conditioning Prevalence Rate'
col = 'case_growth_rate'
source = 'Source: ZTRAX \nGrowth Rate = no AC / total number of properties'
vmin = prevalenceus['ACpercent'].min()
vmax = prevalenceus['ACpercent'].max()
cmap = 'Reds'

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 8))

# Remove the axis
ax.axis('off')

prevalenceus.plot(column='ACpercent', ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap, missing_kwds={'color': 'lightgrey'})

# Add a title
ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
ax.set_xlim([-125,-65])
ax.set_ylim([25,50])
# Create an annotation for the data source
ax.annotate(source, xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', 
            verticalalignment='bottom', fontsize=10)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

# Empty array for the data range
sm._A = []

# Add the colorbar to the figure
cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)

#completeness of AC 

completeDF = DF.loc[DF.AirConditioningTypeorSystemStndCode.notna()].groupby(['FIPS'])['ACpercent'].sum().reset_index()
completeus = usa.merge(completeDF, on =['FIPS'], how='left')
# Initialize the figure

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

title = 'Completeness of Air Conditioning Type'
col = 'case_growth_rate'
source = 'Source: ZTRAX \nGrowth Rate = number of properties have AC info / total number of properties'
vmin = completeus['ACpercent'].min()
vmax = completeus['ACpercent'].max()
cmap = 'Purples'

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(20, 8))

# Remove the axis
ax.axis('off')

completeus.plot(column='ACpercent', ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap, missing_kwds={'color': 'lightgrey'})

# Add a title
ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
ax.set_xlim([-125,-65])
ax.set_ylim([25,50])
# Create an annotation for the data source
ax.annotate(source, xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', 
            verticalalignment='bottom', fontsize=10)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

# Empty array for the data range
sm._A = []

# Add the colorbar to the figure
cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)
