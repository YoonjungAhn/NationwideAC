

# Nationwide AC Project

This repository contains code and documentation for estimating air conditioning prevalence using various data sources and methodologies.

New to XGBoost? The XGBoost documentation has a good [introduction to boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) that explains the type of model used in this project.


## Setup Instructions

1. Update any file paths labeled as `"<your_path_here>"` in the scripts. A comment above each one describes the file expected there (content, source and file type).
2. The R Markdown files (`.Rmd`) use the folder they are saved in as the working directory, so their input and output files are written as relative paths (e.g. `"./Data/Allcombine.csv"`). Put the input files next to the `.Rmd` file or edit those relative paths.
3. Ensure the required R and Python packages are installed.
4. Run each script in the order below if doing full pipeline analysis.

### Pipeline order

| Step | Script | Main input | Main output |
|---|---|---|---|
| 1 | `Dewey_imputation.Rmd` | `Dewey_selected_data.csv` (Dewey property records) | `Dewey_categoryvalues_imputed_v2.csv`, `Dewey_numervalues_imputed.csv` (join by `PROPID` → `Dewey_imputed.csv`) |
| 2 | `Dataintegration_ACestimation.Rmd` | Dewey data + external data sources below | `allcombined_for_contemporal_prediction.csv` |
| 3 | `Yes_ACesimtation_HPC.py` | `allcombined_for_contemporal_prediction.csv` | `yes_ac_predict_result.csv` |
| 4 | `Alltype_ACesimtation_HPC_predict.py` | `allcombined_for_contemporal_prediction.csv`, `yes_ac_predict_result.csv`, `cityruralindex_2020.csv` | `alltype_ac_predict_result.csv`, `prediction_probabilities.csv` |
| 5 | `ACprob_to_county_zip.py` | `alltype_ac_predict_result.csv`, NHGIS and Census boundary shapefiles | Tract, ZCTA, block group, county and metro summaries |
| 6 | `UShouse.Rmd` | Summaries from step 5, Census housing-unit totals | Summaries adjusted to housing-unit totals (see `Data/`) |
| 7 | `Comparison_analysis.py` | Tract summaries from step 5, validation data | Validation figures and correlations |
| 8 | `ResultACmapping.py`, then `sudo_rural_urban_ACmap.Rmd` | Tract summaries from step 5, boundary shapefiles | Maps, urban/rural figures and state summary |


## Data Sources

See Table 2 of the [paper](https://www.nature.com/articles/s41597-025-06104-3) for full details. Dewey and ZTRAX are proprietary and cannot be shared in this repository.

| Data | Variables used | Spatial unit | Source | Used in |
|---|---|---|---|---|
| Dewey property records (proprietary) | AC type, heating type, year built, renovation year, total rooms, condition, owner occupancy, location | Property | [Dewey](https://www.deweydata.io/) | `Dewey_imputation.Rmd`, `Dataintegration_ACestimation.Rmd` |
| Cooling degree days (CDDs) | Annual CDDs, 1970–2024 (included as `Data/Allcombine.csv`) | County | [NOAA NCEI Climate at a Glance](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/time-series) | `Dataintegration_ACestimation.Rmd` |
| American Community Survey 5-year estimates | Median income, % Black, % Hispanic, % higher education | Census tract | [U.S. Census Bureau](https://www.census.gov/programs-surveys/acs) (via [`tidycensus`](https://walker-data.com/tidycensus/)) | `Dataintegration_ACestimation.Rmd` |
| Built-up property records (BUPR) | BUPR | Census tract | [HISDAC-US](https://dataverse.harvard.edu/dataverse/hisdacus) | `Dataintegration_ACestimation.Rmd` |
| Historic Redlining Score | HRI2020 | Census tract (2020) | [openICPSR](https://www.openicpsr.org/openicpsr/project/141121) | `Dataintegration_ACestimation.Rmd` |
| Core Based Statistical Areas | Metro / micro / not metro | CBSA | [Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | `Dataintegration_ACestimation.Rmd`, `ACprob_to_county_zip.py` |
| Tract, ZCTA and block group boundaries (2000, 2010, 2015, 2020); 2020→2010 tract crosswalk | Boundaries | Tract, ZCTA, block group | [NHGIS](https://www.nhgis.org) | `ACprob_to_county_zip.py`, `UShouse.Rmd` |
| Decennial Census / ACS housing units | Total housing units | Tract, county, block group, ZCTA | [U.S. Census Bureau](https://data.census.gov) (via `tidycensus`) | `UShouse.Rmd` |
| American Housing Survey | AC prevalence (validation) | Metro / census division | [U.S. Census Bureau AHS](https://www.census.gov/programs-surveys/ahs.html) | `Comparison_analysis.py` |
| ZTRAX (proprietary, exploratory only) | AC type | Property | Zillow Transaction and Assessment Database | `00.summary.py` |

**NHGIS downloads:** NHGIS names each download with your own extract number (e.g. `nhgis0075_shapefile_tl2020_us_tract_2020`). Our tract extract was `nhgis0075`, our ZCTA extract was `nhgis0077` and our block group extract was `nhgis0080`. Rename the folders in `ACprob_to_county_zip.py` to match your extract.


## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Data Sources](#data-sources)
- [Repository Contents](#repository-contents)
  - [I. Data Preparation and Modeling](#i-data-preparation-and-modeling)
    - [Dewey_imputation.Rmd](#1-dewey_imputationrmd)
    - [Dataintegration_ACestimation.Rmd](#2-dataintegration_acestimationrmd)
    - [Yes_ACesimtation_HPC.py](#3-yes_acesimtation_hpcpy)
    - [Alltype_ACesimtation_HPC_predict.py](#4-alltype_acesimtation_hpc_predictpy)
  - [II. Validation](#ii-validation)
    - [Comparison_analysis.py](#1-comparison_analysispy)
  - [III. Figures](#iii-figures)
    - [sudo_rural_urban_ACmap.Rmd](#1-sudo_rural_urban_acmaprmd)
    - [ResultACmapping.py](#2-resultacmappingpy)
  - [IV. Data Creation](#iv-data-creation)
    - [ACprob_to_county_zip.py](#1-acprob_to_county_zippy)
    - [UShouse.Rmd](#2-ushousermd)
  - [V. Exploratory Analysis](#v-exploratory-analysis)
    - [00.summary.py](#1-00summarypy)

## Repository Contents
## I. Data preparation and modeling  

### 1. `Dewey_imputation.Rmd`
**Purpose**:  
Imputes missing housing attributes in the Dewey property records using random forest imputation (`missForest`).

**Key Features**:
- Tunes `missForest` (`mtry`, `ntree`) on a complete subset with randomly introduced missing values
- Imputes categorical variables (classification, heating type, condition, owner occupancy) and numeric variables (year built, renovation year, total rooms) in parallel chunks

**Dependencies**:
- `dplyr`, `readr`, `tidyr`, `doParallel`, `doRNG`, `foreach`, `missForest`, `data.table`, `ggplot2`

**Output**:  
`Dewey_categoryvalues_imputed_v2.csv` and `Dewey_numervalues_imputed.csv`

---

### 2. `Dataintegration_ACestimation.Rmd`
**Purpose**:  
Merges multiple external and internal datasets to construct the final modeling dataset for AC prevalence prediction.

**Key Features**:
- Loads cleaned and imputed datasets (e.g., Dewey, climate, demographic)
- Performs key joins and harmonizes spatial units (tract, county, ZIP)
- Calculates derived metrics such as degree-days and building characteristics
- Performs quality control checks and produces summary statistics

**Dependencies**:
- `sf`, `dplyr`, `tidyr`, `data.table`, `readr`, `tidycensus`

**Input**:  
See the [Data Sources](#data-sources) table. The file names expected for each input are listed at the top of the file.

**Output**:  
`allcombined_for_contemporal_prediction.csv`: a unified dataframe with one row per property, ready for machine learning modeling.

---

### 3. `Yes_ACesimtation_HPC.py`
**Purpose**:  
Executes the machine learning model to estimate the type of AC for properties recorded only as "YES" (AC present, type unknown) across the U.S., optimized for high-performance computing (HPC) environments.

**Key Features**:
- Trains an XGBoost model using cross-validation
- Applies stratified sampling to maintain spatial representation
- Saves model outputs and prediction scores to disk
- Logs training diagnostics and feature importances

**Dependencies**:
- `pandas`, `numpy`, `xgboost`, `scikit-learn`, `hyperopt`, `shap`, `joblib`, `matplotlib`, `seaborn`

**Input**:  
`allcombined_for_contemporal_prediction.csv` (output of `Dataintegration_ACestimation.Rmd`)

**Output**:  
- `yes_ac_predict_result.csv` (predictions)
- Accuracy and tuning results (`.csv`, `.png`)
- SHAP values (for model interpretation)

---

### 4. `Alltype_ACesimtation_HPC_predict.py`
**Purpose**:  
Generalized version of the estimation script to predict all types of air conditioning systems (e.g., central, evaporative cooler, others, no AC) using the same core pipeline as the YES AC script.

**Key Features**:
- Uses the YES AC predictions to fill in properties recorded as "YES"
- Includes additional features such as year-built, building condition, and local climate metrics

**Dependencies**:
Same as `Yes_ACesimtation_HPC.py`

**Input**:  
`allcombined_for_contemporal_prediction.csv`, `yes_ac_predict_result.csv`, and `cityruralindex_2020.csv` (tract-level urban/rural index used to report accuracy separately for urban and rural tracts)

**Output**:  
- `alltype_ac_predict_result.csv` (predicted AC type for every property)
- `prediction_probabilities.csv` (predicted probability of each AC type, used for the uncertainty layer)
- Accuracy reports overall and by urban/rural, and a confusion matrix


## II. Validation 
### 1. `Comparison_analysis.py`
Compares the predicted AC prevalence with other data sources. For each comparison it draws a scatter plot (other source vs. prediction) and prints the Pearson correlation by AC type:
1. American Housing Survey (AHS), by metro area
2. AHS, national, by census division (all tracts, and non-metro tracts only)
3. Sera et al. (2020), central AC by metro area
4. Romitti et al. (2022), any AC by metro area
5. NYC Housing and Vacancy Survey 2023 (data preparation only)

Set `DATA_DIR` and `FIG_DIR` at the top of the file; every input file is listed in `PATHS`.

## III. Figures
### 1. `sudo_rural_urban_ACmap.Rmd`
An R Markdown document that shows the share of each AC type in urban and rural tracts for every state, with the state panels arranged like a map of the U.S. (`geofacet`). It:
- Reads `results/urban_rural_AC_tractlevel_2020.csv`, written by `ResultACmapping.py` (section 3)
- Sums the tract counts by state, AC type and urbanicity, and converts them to percentages
- Saves `figures/UrbanRuralSudoMap.png`
- Requires `dplyr`, `tidyr`, `ggplot2` and `geofacet`

### 2. `ResultACmapping.py`
Script for the AC maps and summary figures. It:
- Maps the tract-level AC types for ten large metro areas (with and without downtown markers)
- Maps the county-level share of each AC type for the continental U.S.
- Compares urban and rural tracts (urban = tract that intersects a metro area) with a box plot, and writes `urban_rural_AC_tractlevel_2020.csv` for `sudo_rural_urban_ACmap.Rmd`
- Writes a state summary table of AC types
- Uses `geopandas`, `matplotlib` and `seaborn`

## IV. Data creation
### 1. `ACprob_to_county_zip.py`
Aggregates the property-level predictions to census geographies. Each step can be turned on or off at the top of the file:
1. Counts properties by AC type in each census tract, ZIP Code Tabulation Area (ZCTA), metro area (CBSA) and block group, for the 2000, 2010, 2015 and 2020 boundaries (spatial join of property points to boundary shapefiles)
2. Builds a ZCTA-to-county key and adds the county to the ZCTA summaries
3. Formats the tract summaries (county FIPS, tract GEOID, AC type, percentage, count)
4. Builds a tract-to-metro-area key, and (after `UShouse.Rmd`) adds the metro area to the adjusted tract data

Uses NHGIS tract, ZCTA and block group shapefiles and Census CBSA, county and state boundaries (see [Data Sources](#data-sources)).

### 2. `UShouse.Rmd`
An R Markdown document that adjusts the predicted AC counts to match Census housing-unit totals. Run it after `ACprob_to_county_zip.py`. It:
- Downloads total housing units by tract, county and block group with `tidycensus` (requires a [Census API key](https://api.census.gov/data/key_signup.html))
- Uses the NHGIS 2020→2010 tract crosswalk to align housing units with 2010 tract boundaries
- Proportionally scales the predicted counts so they do not exceed the number of housing units in each tract, county, block group and ZCTA
- Requires `tidycensus` and `tidyverse`

## V. Exploratory analysis
### 1. `00.summary.py`
An early exploratory script (not part of the final pipeline) that summarizes the raw ZTRAX property records by county and maps the AC missing rate, AC prevalence, no-AC rate, and completeness of the AC type field.


## Citation
Ahn. Y & Uejio. C "A Comprehensive Dataset of Residential Air Conditioning Prevalence in the Continental United States" https://www.nature.com/articles/s41597-025-06104-3
