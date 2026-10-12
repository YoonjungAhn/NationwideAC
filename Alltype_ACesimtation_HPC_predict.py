#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:20:32 2024

@author: y943a214

Step 4 of the pipeline: predict the AC type of every property
(Central, Evaporative Cooler, NoAC, Others) with an XGBoost classifier,
including properties whose AC information is missing.

Inputs
------
allcombined_for_contemporal_prediction.csv
    Property-level modeling table, output of Dataintegration_ACestimation.Rmd.
yes_ac_predict_result.csv
    Output of Yes_ACesimtation_HPC.py (predicted AC type, column "Yespredicted").
cityruralindex_2020.csv
    Tract-level urban/rural index (GEOID, cityrural = 'Urban' or 'Rural'), used to
    report accuracy separately for urban and rural tracts.

Outputs (written to the working directory)
------------------------------------------
alltype_ac_predict_result.csv                  predicted AC type for every property
                                               ("finalAIRCONDITIONINGpre"); input of ACprob_to_county_zip.py
prediction_probabilities.csv                   predicted probability of each AC type (uncertainty layer)
AlltypesAC_predict_accuracy_matrix.csv         precision / recall / F1 on the test set
AlltypesAC_urbanrural_predict_accuracy_matrix.csv  the same, for urban and rural tracts
confusion_matrix_allAC.csv / .png              confusion matrix on the test set

New to XGBoost? See https://xgboost.readthedocs.io/en/stable/tutorials/model.html
SHAP references:
  https://www.kaggle.com/code/bennyfung/model-interpretability-xgboost-shap
  https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.html
"""

import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn import metrics
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# 0. Settings
# =============================================================================
# Folder that holds the input CSVs; outputs are written here too
os.chdir('<your_path_here>')

RUN_TUNING = False  # True: re-run the hyperopt search (slow). False: use the tuned values in section 5
RUN_SHAP = False    # True: compute SHAP values for each county (slow)

N_JOBS = multiprocessing.cpu_count()  # use all available cores


# =============================================================================
# 1. Load data
# =============================================================================
# Urban/rural index by census tract (11-digit GEOID, zero-padded)
city = pd.read_csv('./cityruralindex_2020.csv')
city.GEOID = city.GEOID.apply(lambda x: '{0:0>11}'.format(x))

# YES AC predictions from Yes_ACesimtation_HPC.py: use the predicted type as the AC type
predictdf = pd.read_csv('./yes_ac_predict_result.csv')
predictdf['AIRCONDITIONINGrecode'] = predictdf['Yespredicted']

# Property-level modeling table from Dataintegration_ACestimation.Rmd
df = pd.read_csv('./allcombined_for_contemporal_prediction.csv', na_values=[''], dtype={9: str}, low_memory=False)
df.GEOID = df.GEOID.apply(lambda x: '{0:0>11}'.format(x))
df = df.drop(columns=['prop_poverty', 'prop_renter', 'Metorlevel'])  # not used in the model


# =============================================================================
# 2. Prepare data
# =============================================================================
# AIRCONDITIONINGCODE 6 is recoded as no AC
df.loc[df['AIRCONDITIONINGCODE'] == 6, 'AIRCONDITIONINGrecode'] = 'NoAC'

# Keep the continental U.S. only
df = df.loc[~df['STATE'].isin(['AK', 'HI', 'PR'])]

# Imputed values can be fractional; round year built and number of rooms
df['YEARBUILT'] = df['YEARBUILT'].round()
df['TOTROOMS'] = df['TOTROOMS'].round()

# Drop properties without heating type, then one-hot encode it
# (REHEAT_Central, REHEAT_HeatPump, REHEAT_Other)
df = df.dropna(subset=['REHEATTYPE'])
df = pd.get_dummies(df, columns=['REHEATTYPE'], prefix='REHEAT', drop_first=False)

# Replace the "Yes" (AC present, type unknown) rows with the rows from the YES AC model.
# predictdf is reduced to the same columns as df so the two tables can be stacked.
df = df.loc[df['AIRCONDITIONINGrecode'] != 'Yes']
predictdf = predictdf.drop(columns=[col for col in predictdf.columns if col not in df.columns])
for col in [c for c in df.columns if c not in predictdf.columns]:
    predictdf[col] = np.nan
predictdf = predictdf[df.columns]
df = pd.concat([df, predictdf], axis=0, ignore_index=True)

# Treat the text 'nan' as missing
df['AIRCONDITIONINGrecode'] = df['AIRCONDITIONINGrecode'].replace('nan', np.nan)

# Add the urban/rural index
df = pd.merge(df, city, how='left', on='GEOID')

# DF keeps every property (including missing AC type) for the final prediction (section 7);
# df keeps only properties with a known AC type for training and testing.
DF = df.copy()
df = df.loc[df['AIRCONDITIONINGrecode'].notna()]

# Encode the AC type as integers for XGBoost
# (alphabetical: 0 Central, 1 Evaporative Cooler, 2 NoAC, 3 Others)
label_encoder_air = LabelEncoder()
df['AIRCONDITIONINGrecode_encoded'] = label_encoder_air.fit_transform(df['AIRCONDITIONINGrecode'].astype(str))
# NOTE: the encoder is re-fitted on DF, which also contains missing values ('nan' -> code 4).
# Codes 0-3 stay the same, so predictions are not affected.
DF['AIRCONDITIONINGrecode_encoded'] = label_encoder_air.fit_transform(DF['AIRCONDITIONINGrecode'].astype(str))

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
        'max_depth': hp.choice('max_depth', range(3, 18)),
        'gamma': hp.uniform('gamma', 1, 9),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10, 1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
    }

    def objective(space):
        """Train one candidate model and return its (negative) test accuracy."""
        clf = xgb.XGBClassifier(
            n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
            reg_alpha=space['reg_alpha'], min_child_weight=int(space['min_child_weight']),
            colsample_bytree=space['colsample_bytree'], enable_categorical=True,
            tree_method="hist", n_jobs=N_JOBS)
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        print("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK}

    trials = Trials()
    tuned = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)
    print("The best hyperparameters are:", tuned)

    # Save every trial and plot accuracy against each hyperparameter
    params = pd.DataFrame(trials.vals)
    params['accuracy'] = [-r['loss'] for r in trials.results]
    params.to_csv('./alltypeAC_model_training_trial_results.csv')

    params_to_plot = ['max_depth', 'n_estimators', 'gamma', 'reg_alpha', 'min_child_weight', 'colsample_bytree']
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for ax, param in zip(axes.flatten(), params_to_plot):
        params.groupby(param)['accuracy'].mean().sort_index().plot(
            ax=ax, title=f'Accuracy per {param}', legend=False, marker='o')
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Accuracy')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('./AlltypesAC_predict_parameter_tuning.png', bbox_inches='tight')
    plt.show()


# =============================================================================
# 5. Final model with the tuned hyperparameters
# =============================================================================
best_hyperparams = {'colsample_bytree': 0.5380968316576427, 'gamma': 2.8641487502043637, 'max_depth': 14,
                    'min_child_weight': 0, 'n_estimators': 580, 'reg_alpha': 0.3784907977762958}

best_model = xgb.XGBClassifier(
    **best_hyperparams,
    enable_categorical=True,
    objective='multi:softprob',   # multi-class model that outputs a probability for each class
    random_state=42,
    tree_method="hist",           # faster training on large data
    n_jobs=N_JOBS
)
best_model.fit(X_train, y_train)


# =============================================================================
# 6. Evaluate on the test set
# =============================================================================
y_pred = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# 6a. Classification report with AC type names instead of codes
original_classes = label_encoder_air.inverse_transform([0, 1, 2, 3])
class_name_map = dict(zip([str(i) for i in range(len(original_classes))], original_classes))

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().reset_index()
report_df['index'] = report_df['index'].replace(class_name_map)
report_df.rename(columns={'index': 'AC_Type'}, inplace=True)
print(report_df)
report_df.to_csv('./AlltypesAC_predict_accuracy_matrix.csv', index=False)

# 6b. Classification report for urban and rural tracts separately
eval_df = pd.DataFrame({
    'true': label_encoder_air.inverse_transform(y_test),
    'pred': label_encoder_air.inverse_transform(y_pred),
    'urbanicity': df.loc[y_test.index, 'cityrural']
})

urbanrural_reports = []
for group in ['Urban', 'Rural']:
    print(f"Classification Report for {group} Tracts:")
    subset = eval_df[eval_df['urbanicity'] == group]
    group_report = pd.DataFrame(classification_report(
        subset['true'], subset['pred'],
        labels=label_encoder_air.classes_, output_dict=True, zero_division=0)).transpose()
    print(group_report.round(2))
    group_report['urbanicity'] = group
    urbanrural_reports.append(group_report)

pd.concat(urbanrural_reports).to_csv('./AlltypesAC_urbanrural_predict_accuracy_matrix.csv')

# 6c. Confusion matrix (rows = true AC type, columns = predicted AC type)
all_labels = label_encoder_air.classes_
cm = confusion_matrix(label_encoder_air.inverse_transform(y_test),
                      label_encoder_air.inverse_transform(y_pred), labels=all_labels)
cm_df = pd.DataFrame(cm,
                     index=[f"True: {label}" for label in all_labels],
                     columns=[f"Pred: {label}" for label in all_labels])
cm_df.to_csv('./confusion_matrix_allAC.csv', index=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('./confusion_matrix_allAC.png', dpi=300)
plt.close()


# =============================================================================
# 7. Predict the AC type of every property and save results
# =============================================================================
X_original = DF[features]
y_original = DF[target]

original_predictions = best_model.predict(X_original)

# NOTE: y_original also contains properties with a missing AC type (code 4),
# so this accuracy is lower than the test-set accuracy in section 6.
original_accuracy = accuracy_score(y_original, original_predictions)
print("Original Data Accuracy:", original_accuracy)

# Predicted AC type as a name (column "finalAIRCONDITIONINGpre")
DF['final_prediction'] = original_predictions
DF['finalAIRCONDITIONINGpre'] = label_encoder_air.inverse_transform(DF['final_prediction'])

# Property-level results; this file is the input of ACprob_to_county_zip.py
DF.to_csv('./alltype_ac_predict_result.csv', index=False)

# Predicted probability of each AC type (one column per type: prob_Central, prob_NoAC, ...)
probs = best_model.predict_proba(X_original)
class_labels = label_encoder_air.inverse_transform(np.arange(probs.shape[1]))
prob_df = pd.DataFrame(probs, columns=[f"prob_{label}" for label in class_labels], index=X_original.index)

# Property-level probabilities, used for the uncertainty layer in Data/
id_cols = ['GEOID', 'GEOIDCN', 'PROPID', 'LAT', 'LON', 'COUNTY', 'STATE']
pd.concat([DF[id_cols], prob_df], axis=1).to_csv("./prediction_probabilities.csv", index=False)


# =============================================================================
# 8. (Optional) SHAP values by county, to show which predictors drive the model
# =============================================================================
if RUN_SHAP:
    explainer = shap.Explainer(best_model)

    result_dfs = []
    for fips_code, group in df.groupby('GEOIDCN'):   # GEOIDCN = 5-digit county FIPS
        shap_values_county = explainer(group[features])

        # Multi-class output has shape (rows, features, classes): keep the first class
        if len(shap_values_county.shape) == 3:
            shap_values_county = shap_values_county[:, :, 0]

        # One row per property: original columns + one SHAP value per feature
        shap_values_df = pd.DataFrame(shap_values_county.values, columns=features)
        shap_values_df['PROPID'] = group['PROPID'].values
        result_df = pd.concat([group.reset_index(drop=True), shap_values_df], axis=1)
        result_dfs.append(result_df)

        result_df.to_csv(f"./shap_values_county_{fips_code}.csv", index=False)
        print(f"Saved SHAP values for county {fips_code}")

    pd.concat(result_dfs).to_csv("./shap_values_county_level_all.csv", index=False)
