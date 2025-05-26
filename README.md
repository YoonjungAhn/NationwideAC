

# Nationwide AC Project

This repository contains code and documentation for estimating air conditioning prevalence using various data sources and methodologies.


## Setup Instructions

1. Update any file paths labeled as `"<your_path_here>"` in the scripts.
2. Ensure the required R and Python packages are installed.
3. Run each script in the intended order if doing full pipeline analysis.


## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Repository Contents](#repository-contents)
  - [I. Data Preparation and Modeling](#i-data-preparation-and-modeling)
    - [Dataintegration_ACestimation.Rmd](#1-dataintegration_acestimationrmd)
    - [Yes_ACesimtation_HPC.py](#2-yes_acesimtation_hpcpy)
    - [Alltype_ACesimtation_HPC.py](#3-alltype_acesimtation_hpcpy)
  - [II. Validation](#ii-validation)
    - [Comaprison_analysis.py](#1-comaprison_analysispy)
  - [III. Figures](#iii-figures)
    - [sudo_rural_urban_ACmap.Rmd](#1-sudo_rural_urban_acmaprmd)
    - [ResultACmapping.py](#2-resultacmappingpy)
  - [IV. Data Creation](#iv-data-creation)
    - [cleaned_AC_to_county_zip.py](#1-cleaned_ac_to_county_zippy)

## Repository Contents
## I. Data preparation and modeling  

### 1. `Dataintegration_ACestimation.Rmd`
**Purpose**:  
Merges multiple external and internal datasets to construct the final modeling dataset for AC prevalence prediction.

**Key Features**:
- Loads cleaned and imputed datasets (e.g., Dewey, climate, demographic)
- Performs key joins and harmonizes spatial units (tract, county, ZIP)
- Calculates derived metrics such as degree-days and building characteristics
- Performs quality control checks and produces summary statistics

**Dependencies**:
- `sf`, `dplyr`, `lubridate`, `data.table`, `tigris`, `tidycensus`

**Output**:  
A unified dataframe with one row per spatial unit (e.g., property or tract), ready for machine learning modeling.

---

### 2. `Yes_ACesimtation_HPC.py`
**Purpose**:  
Executes the machine learning model to estimate the prevalence of YES-type air conditioning systems across the U.S., optimized for high-performance computing (HPC) environments.

**Key Features**:
- Trains an XGBoost model using cross-validation
- Applies stratified sampling to maintain spatial representation
- Saves model outputs and prediction scores to disk
- Logs training diagnostics and feature importances

**Dependencies**:
- `pandas`, `xgboost`, `scikit-learn`, `joblib`, `argparse`

**Input**:  
Preprocessed dataset (from RMarkdown output) in CSV or pickle format

**Output**:  
- Trained model (`.joblib`)
- Predictions (`.csv`)
- SHAP values (optional, for model interpretation)

---

### 3. `Alltype_ACesimtation_HPC.py`
**Purpose**:  
Generalized version of the estimation script to predict all types of air conditioning systems (e.g., central, window, any AC) using the same core pipeline as the YES AC script.

**Key Features**:
- Flexible model training using `argparse` to pass target AC type
- Includes additional features such as year-built, building condition, and local climate metrics
- Can be run in batch mode for multiple target variables

**Dependencies**:
Same as `Yes_ACesimtation_HPC.py`

**Output**:  
Multiple prediction files for different AC types, along with diagnostics and trained models.


## II. Validation 
### 1. `Comaprison_analysis.py`
A Python script for comparing AC estimation outputs across different modeling approaches. It:
- Loads prediction results
- Calculates difference metrics (e.g., RMSE, MAE)
- Visualizes model comparison with summary statistics or graphs
- Designed for reproducible model evaluation

## III. Figures
### 1. `sudo_rural_urban_ACmap.Rmd`
An R Markdown document for visualizing rural and urban disparities in AC availability using statistical maps. It:
- Loads spatial and model output data
- Categorizes counties by urban/rural typologies
- Generates comparative maps for interpretation
- Requires R packages such as `tidyverse`, `sf`, `tmap`, and `ggplot2`

### 2. `ResultACmapping.py`
Script for generating final AC mapping visualizations. It:
- Integrates model predictions with geographic boundaries
- Produces choropleth maps by county or ZIP code
- Utilizes libraries like `geopandas`, `matplotlib`, and `contextily`

## IV. Data creation
### 1. `cleaned_AC_to_county_zip.py`
A data processing utility that:
- Maps estimated AC prevalence from census tract or block group level to ZIP Code Tabulation Areas (ZCTAs) and counties
- Performs spatial joins and weighted aggregation
- Includes support for custom shapefiles and weighting schemes


## Citation
Ahn. Y & Uejio. C "A Comprehensive Dataset of Residential Air Conditioning Prevalence in the Continental United States"

