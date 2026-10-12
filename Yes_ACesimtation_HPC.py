#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:20:32 2024

@author: y943a214

Step 3 of the pipeline: predict the AC type (Central, Evaporative Cooler,
Others) for properties whose AC record only says "Yes" (AC present, type
unknown), using an XGBoost classifier.

Input
-----
allcombined_for_contemporal_prediction.csv
    Property-level modeling table, output of Dataintegration_ACestimation.Rmd.

Outputs (written to the working directory)
------------------------------------------
yes_ac_predict_result.csv            properties with the predicted AC type ("Yespredicted");
                                     input of Alltype_ACesimtation_HPC_predict.py
YesAC_predict_accuracy_matrix.csv    precision / recall / F1 on the test set
original_accuracy_results.csv        accuracy on all rows used for prediction
YesAC_predict_shap_values_<county>.pkl, results/yes_AC_county_summary/*.csv
                                     SHAP values by county (if RUN_SHAP is True)

New to XGBoost? See https://xgboost.readthedocs.io/en/stable/tutorials/model.html
SHAP references:
  https://www.kaggle.com/code/bennyfung/model-interpretability-xgboost-shap
  https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.html
"""

import glob
import multiprocessing
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# 0. Settings
# =============================================================================
# Folder that holds allcombined_for_contemporal_prediction.csv; outputs are written here too
os.chdir('<your_path_here>')

RUN_TUNING = False      # True: re-run the hyperopt search (slow). False: use the tuned values in section 5
RUN_SHAP = True         # True: compute SHAP values for each county (slow)
RUN_SHAP_PLOTS = False  # True: combine saved SHAP files and draw a summary plot

# Leave a few cores free for other jobs on the HPC node
N_JOBS = max(1, multiprocessing.cpu_count() - 8)


# =============================================================================
# 1. Load data
# =============================================================================
# Property-level modeling table: output of Dataintegration_ACestimation.Rmd
df = pd.read_csv('./allcombined_for_contemporal_prediction.csv', na_values=[''])

# Keep the continental U.S. only
df = df.loc[~df['STATE'].isin(['AK', 'HI', 'PR'])]

# Imputed values can be fractional; round year built and number of rooms
df['YEARBUILT'] = round(df['YEARBUILT'])
df['TOTROOMS'] = round(df['TOTROOMS'])

# Correlation between the candidate numeric predictors
corr_matrix = df[['YEARBUILT', 'RENOYEAR', 'TOTROOMS', 'BUPR', 'median_income', 'prop_Black',
                  'prop_All_Spanish', 'prop_HigherEdu', 'CDDs', 'HRI2020']].corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# =============================================================================
# 2. Prepare data
# =============================================================================
# Drop properties without heating type, keep the heating type label for the
# output (column HEAT), then one-hot encode it (REHEAT_Central, REHEAT_HeatPump, REHEAT_Other)
df = df.dropna(subset=['REHEATTYPE'])
df['HEAT'] = df['REHEATTYPE']
df = pd.get_dummies(df, columns=['REHEATTYPE'], prefix='REHEAT', drop_first=False)

# "Yes" means AC is present but the type is unknown: set it to missing.
# Keep only properties with a known AC type other than NoAC
# (Central, Evaporative Cooler, Others).
df['AIRCONDITIONINGrecode'] = df['AIRCONDITIONINGrecode'].replace("Yes", np.nan)
df = df[(df['AIRCONDITIONINGrecode'] != 'NoAC') & (df['AIRCONDITIONINGrecode'].notna())]

# Encode the AC type as integers for XGBoost (alphabetical: 0 Central, 1 Evaporative Cooler, 2 Others)
label_encoder_air = LabelEncoder()
df['AIRCONDITIONINGrecode_encoded'] = label_encoder_air.fit_transform(df['AIRCONDITIONINGrecode'].astype(str))

# Cooling degree day quintiles, used only to stratify the train/test split
df['CDDs_quintiles'] = pd.qcut(df['CDDs'], 5, labels=[1, 2, 3, 4, 5])

# Model predictors and target
features = ['RENOYEAR', 'TOTROOMS', 'CONDITION', 'REHEAT_Central', 'REHEAT_HeatPump', 'REHEAT_Other',
            'BUPR', 'median_income', 'prop_Black', 'prop_All_Spanish', 'prop_HigherEdu', 'CDDs', 'HRI2020']
target = 'AIRCONDITIONINGrecode_encoded'


# =============================================================================
# 3. Train/test split (70/30), stratified by CDD quintile x AC type
# =============================================================================
df['stratify_col'] = df['CDDs_quintiles'].astype(str) + "_" + df['AIRCONDITIONINGrecode']
df['stratify_col'] = df['stratify_col'].fillna('Missing')

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['stratify_col'])

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]


# =============================================================================
# 4. (Optional) Hyperparameter tuning with hyperopt
#    Guide: https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
# =============================================================================
if RUN_TUNING:
    space = {
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'max_depth': hp.choice('max_depth', range(3, 15)),
        'gamma': hp.uniform('gamma', 0, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
    }

    def objective(space):
        """Train one candidate model and return its (negative) test accuracy."""
        clf = xgb.XGBClassifier(
            n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
            reg_alpha=space['reg_alpha'], min_child_weight=int(space['min_child_weight']),
            colsample_bytree=space['colsample_bytree'], enable_categorical=True, n_jobs=N_JOBS)
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    trials = Trials()
    tuned = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    print("The best hyperparameters are:", tuned)

    # Save every trial and plot accuracy against each hyperparameter
    params = pd.DataFrame(trials.vals)
    params['accuracy'] = [-r['loss'] for r in trials.results]
    params.to_csv('./model_training_trial_results.csv')

    params_to_plot = ['max_depth', 'n_estimators', 'gamma', 'reg_alpha', 'min_child_weight', 'colsample_bytree']
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for ax, param in zip(axes.flatten(), params_to_plot):
        params.groupby(param)['accuracy'].mean().sort_index().plot(
            ax=ax, title=f'Accuracy per {param}', legend=False, marker='o')
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Accuracy')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('./YesAC_predict_parameter_tuning.png', bbox_inches='tight')
    plt.show()


# =============================================================================
# 5. Final model with the tuned hyperparameters
# =============================================================================
# Tuning results were stored as positions in the lists below; translate them to values
colsample_bytree_values = np.arange(0.6, 1.05, 0.05).tolist()
max_depth_range = list(range(5, 15))
n_estimators_range = list(range(200, 1001, 100))
min_child_weight_range = list(range(1, 10))

best_hyperparams = {
    'colsample_bytree': colsample_bytree_values[8],   # 1.0
    'gamma': 0.7482189838249866,
    'max_depth': max_depth_range[0],                  # 5
    'min_child_weight': min_child_weight_range[6],    # 7
    'n_estimators': n_estimators_range[3],            # 500
    'reg_alpha': 0.7600200477918414
}

best_model = xgb.XGBClassifier(
    **best_hyperparams,
    enable_categorical=True,
    objective='multi:softprob',   # multi-class model that outputs a probability for each class
    random_state=42,
    n_jobs=N_JOBS
)
best_model.fit(X_train, y_train)


# =============================================================================
# 6. Evaluate on the test set
# =============================================================================
y_pred = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().reset_index()
report_df.to_csv('./YesAC_predict_accuracy_matrix.csv', index=False)


# =============================================================================
# 7. Predict and save results
# =============================================================================
# NOTE: df here contains the properties with a known AC type (section 2).
X_original = df[features]
y_original = df[target]

original_predictions = best_model.predict(X_original)
df['predictions'] = original_predictions
df['Yespredicted'] = label_encoder_air.inverse_transform(df['predictions'])
print(df.head())

original_accuracy = accuracy_score(y_original, original_predictions)
print("Original Data Accuracy:", original_accuracy)
pd.DataFrame({'Original_Accuracy': [original_accuracy]}).to_csv('./original_accuracy_results.csv', index=False)

# Property-level results with the predicted AC type (column "Yespredicted").
# This file is an input of Alltype_ACesimtation_HPC_predict.py.
df.to_csv('./yes_ac_predict_result.csv', index=False)


# =============================================================================
# 8. (Optional) SHAP values by county, to show which predictors drive the model
# =============================================================================
if RUN_SHAP:
    explainer = shap.Explainer(best_model)
    os.makedirs('./results/yes_AC_county_summary', exist_ok=True)

    result_dfs = []
    for fips_code, group in df.groupby('GEOIDCN'):   # GEOIDCN = 5-digit county FIPS
        X_county = group[features]
        shap_values_county = explainer(X_county)

        # Save the SHAP Explanation object for the county (used in section 9)
        with open(f'./YesAC_predict_shap_values_{fips_code}.pkl', 'wb') as f:
            pickle.dump(shap_values_county, f)

        # Multi-class output has shape (rows, features, classes): keep the first class
        if len(shap_values_county.shape) == 3:
            shap_values_county = shap_values_county[:, :, 0]

        # One row per property: original columns + one SHAP value per feature
        shap_values_df = pd.DataFrame(shap_values_county.values, columns=features)
        shap_values_df['PROPID'] = group['PROPID'].values
        result_df = pd.concat([group.reset_index(drop=True), shap_values_df], axis=1)
        result_dfs.append(result_df)

        result_df.to_csv(f"./results/yes_AC_county_summary/shap_values_county_{fips_code}.csv", index=False)
        print(f'SHAP values saved for FIPS code {fips_code}')

    final_result_df = pd.concat(result_dfs)
    final_result_df.to_csv("./results/yes_AC_county_summary/shap_values_county_level_all.csv", index=False)


# =============================================================================
# 9. (Optional) SHAP summary plot from the saved county files
# =============================================================================
if RUN_SHAP_PLOTS:
    file_paths = glob.glob('./YesAC_predict_shap_values_*.pkl')
    explanations = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            explanations.append(pickle.load(f))

    combined_shap_values = np.vstack([e.values for e in explanations])
    combined_data = np.vstack([e.data for e in explanations])

    # Readable feature names for the plot
    feature_name_mapping = {
        'TOTROOMS': 'Total rooms',
        'CONDITION': 'Condition',
        'RENOYEAR': 'Renovated year',
        'median_income': 'Median income',
        'prop_Black': '%Black/African Americans',
        'prop_All_Spanish': '%Hispanics',
        'prop_HigherEdu': "Post Secondary Education Rate",
        'CDDs': 'Cooling Degree Days',
        'HRI2020': 'Historical Housing Policy Score'
    }
    X_shap = pd.DataFrame(combined_data, columns=features).rename(columns=feature_name_mapping)

    # Summary plot for the first class (index 0 = Central)
    plt.figure(figsize=(10, 10))
    shap.summary_plot(combined_shap_values[:, :, 0], X_shap, feature_names=X_shap.columns.tolist(), show=False)
    plt.savefig('./Local_Importance_summary_plot_with_colorbar.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
