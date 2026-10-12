#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:20:32 2024

@author: y943a214
"""
#https://www.kaggle.com/code/bennyfung/model-interpretability-xgboost-shap
#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.html
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import model_selection, metrics
from xgboost import XGBClassifier
import xgboost as xgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import classification_report
import numpy as np
from joblib import Parallel, delayed
import pickle
import multiprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Set the working directory to the folder that holds the input CSVs below
# (outputs are also written here)
os.chdir('<your_path_here>')

# Tract-level urban/rural index (CSV) with GEOID and cityrural ('Urban'/'Rural'),
# used to report accuracy separately for urban and rural tracts
city = pd.read_csv('./cityruralindex_2020.csv')
city.GEOID =city.GEOID.apply(lambda x: '{0:0>11}'.format(x))


#i need to round up the room numbers
# i need to round up built year
# i need to save shape value as a dataframe
#predictdf =predictdf [['PROPID','Yespredicted']]

# YES AC predictions: output of Yes_ACesimtation_HPC.py
predictdf = pd.read_csv('./yes_ac_predict_result.csv')
predictdf['AIRCONDITIONINGrecode'] = predictdf['Yespredicted']



# Property-level modeling table: output of Dataintegration_ACestimation.Rmd
df = pd.read_csv('./allcombined_for_contemporal_prediction.csv',  na_values=[''], dtype={9: str},low_memory=False)
print(df.columns)
df.GEOID =df.GEOID.apply(lambda x: '{0:0>11}'.format(x))
# Drop specified columns
df = df.drop(columns=['prop_poverty', 'prop_renter', 'Metorlevel'])

df.loc[df['AIRCONDITIONINGCODE'] == 6, 'AIRCONDITIONINGrecode'] = 'NoAC'

df = df.loc[~df['STATE'].isin(['AK', 'HI', 'PR'])]
df['YEARBUILT'] = df['YEARBUILT'].round()
df['TOTROOMS'] = df['TOTROOMS'].round()

df['AIRCONDITIONINGrecode'].unique()
df['REHEATTYPE'].unique()
#DF = df.copy()
df = df.dropna(subset=['REHEATTYPE'])
df = pd.get_dummies(df, columns=['REHEATTYPE'], prefix='REHEAT', drop_first=False)



# Filter out 'Yes' rows
df = df.loc[df['AIRCONDITIONINGrecode'] != 'Yes']

# Drop extra columns from predictdf not in df
extra_columns = [col for col in predictdf.columns if col not in df.columns]
predictdf = predictdf.drop(columns=extra_columns)
#df = pd.merge(df,predictdf, how='left', on = ['PROPID'])

# Fill any df columns missing from predictdf with NaN so concat doesn't drop them
for col in [c for c in df.columns if c not in predictdf.columns]:
    predictdf[col] = np.nan
predictdf = predictdf[df.columns]
df = pd.concat([df, predictdf], axis=0, ignore_index=True)


len(df.loc[df.AIRCONDITIONINGrecode=='NoAC'])
len(df.loc[df.AIRCONDITIONINGrecode=='nan'])

#DF.groupby('Yespredicted')['PROPID'].count()
#DF.groupby('AIRCONDITIONINGrecode')['PROPID'].count()


# Replace 'YES' or NaN values in 'AIRCONDITIONINGrecode' with corresponding values from 'Yespredicted'
df['AIRCONDITIONINGrecode'] = df['AIRCONDITIONINGrecode'].replace('nan', np.nan)



#df['AIRCONDITIONINGcombined'] = np.where(df['AIRCONDITIONINGrecode'].isin(['Yes']), 
#                                       df['Yespredicted'], 
#                                       df['AIRCONDITIONINGrecode'])


#predicting YES AC

#df= df.sample(n=200000, replace=False, random_state=42)

df = pd.merge(df,city, how ='left', on ='GEOID')

DF = df.copy()
len(DF)
#encoding the data
label_encoder_air = LabelEncoder()
#label_encoder_heat = LabelEncoder()



df  = df.loc[df['AIRCONDITIONINGrecode'].notna()]

df['AIRCONDITIONINGrecode_encoded'] = label_encoder_air.fit_transform(df['AIRCONDITIONINGrecode'].astype(str))
#df['HEAT_encoded'] = label_encoder_heat.fit_transform(df['REHEATTYPE'].astype(str))
df['CDDs_quintiles'] = pd.qcut(df['CDDs'], 5, labels=[1, 2, 3, 4, 5])
df['CDDs_quintiles'].astype(str)
df['CONDITION'].astype(str)
#df['HEAT_encoded'].astype(str)

DF['AIRCONDITIONINGrecode_encoded'] = label_encoder_air.fit_transform(DF['AIRCONDITIONINGrecode'].astype(str))
#DF['HEAT_encoded'] = label_encoder_heat.fit_transform(DF['REHEATTYPE'].astype(str))
DF['CDDs_quintiles'] = pd.qcut(DF['CDDs'], 5, labels=[1, 2, 3, 4, 5])
DF['CDDs_quintiles'].astype(str)
DF['CONDITION'].astype(str)
#DF['HEAT_encoded'].astype(str)
DF['stratify_col'] = DF['CDDs_quintiles'].astype(str) + "_" + DF['AIRCONDITIONINGrecode']
DF['stratify_col'] = DF['stratify_col'].fillna('Missing')



# Define the feature columns and target column
features = df[['RENOYEAR', 'TOTROOMS','CONDITION','REHEAT_Central', 'REHEAT_HeatPump', 'REHEAT_Other', 'BUPR','median_income','prop_Black','prop_All_Spanish','prop_HigherEdu','CDDs','HRI2020']].columns# CDDs Adjust indexing if necessary
target = 'AIRCONDITIONINGrecode_encoded'


# Create stratified samples
# We need to ensure that each group (combination of STATE and AIRCONDITIONINGrecode) is represented
# First, create a column for stratification
df['stratify_col'] = df['CDDs_quintiles'].astype(str) + "_" + df['AIRCONDITIONINGrecode']
df['stratify_col'] = df['stratify_col'].fillna('Missing')
# Split the data into train and test data


# Perform stratified split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['stratify_col'])

#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['AIRCONDITIONINGrecode'])

# Separate features and target
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]


#Define the model
xgb_model =xgb.XGBClassifier(objective='multi:softprob', random_state=42, enable_categorical=True)
'''

#find the best parmaters : https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
#Tuneing the model 
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'max_depth': hp.choice('max_depth', range(3, 18)),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'min_child_weight': hp.choice('min_child_weight', range(1, 10,1)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
}

def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']),enable_categorical=True,tree_method="hist" , n_jobs=multiprocessing.cpu_count())
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
                eval_set=evaluation,
                verbose=False)
    
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 5,
                        trials = trials)

params = pd.DataFrame(trials.vals)


params.to_csv('./alltypeAC_model_training_trial_results.csv')


print("The best hyperparameters are: ", "\n")
print(best_hyperparams)

# Define the best model with the best hyperparameters
best_model = xgb.XGBClassifier(
    n_estimators=best_hyperparams['n_estimators'],
    max_depth=int(best_hyperparams['max_depth']),
    gamma=best_hyperparams['gamma'],
    reg_alpha=best_hyperparams['reg_alpha'],
    min_child_weight=int(best_hyperparams['min_child_weight']),
    colsample_bytree=best_hyperparams['colsample_bytree'],
    enable_categorical=True,
    objective='multi:softprob',
    random_state=42, n_jobs = multiprocessing.cpu_count()
)



# this is what i got from the model to predict YES AC

# Hyperparameter value ranges
colsample_bytree_values = np.arange(0.6, 1.05, 0.05).tolist()
max_depth_range = list(range(5, 15))
n_estimators_range = list(range(200, 1001, 100))
min_child_weight_range = list(range(1, 10))
'''


'''

# Translate indices to actual values
best_hyperparams['colsample_bytree'] = colsample_bytree_values[best_hyperparams['colsample_bytree']]
best_hyperparams['max_depth'] = max_depth_range[best_hyperparams['max_depth']]
best_hyperparams['n_estimators'] = n_estimators_range[best_hyperparams['n_estimators']]
best_hyperparams['min_child_weight'] = min_child_weight_range[best_hyperparams['min_child_weight']]

# Setup and train the model
best_model = xgb.XGBClassifier(
    n_estimators=best_hyperparams['n_estimators'],
    max_depth=best_hyperparams['max_depth'],
    gamma=best_hyperparams['gamma'],
    reg_alpha=best_hyperparams['reg_alpha'],
    min_child_weight=best_hyperparams['min_child_weight'],
    colsample_bytree=best_hyperparams['colsample_bytree'],
    enable_categorical=True,
    objective='multi:softprob',
    random_state=42,
    n_jobs=multiprocessing.cpu_count()
)

'''

# this is what i got from the model to predict all types AC
best_hyperparams = {'colsample_bytree': 0.5380968316576427, 'gamma': 2.8641487502043637, 'max_depth': 14, 'min_child_weight': 0, 'n_estimators': 580, 'reg_alpha': 0.3784907977762958}


best_model = xgb.XGBClassifier(
    **best_hyperparams,  # Unpack hyperparameters directly into function arguments
    enable_categorical=True,
    objective='multi:softprob',
    random_state=42,
    tree_method="hist",          # optional but good for speed
    n_jobs=multiprocessing.cpu_count()  # Automatically use all available cores
)



# Fit the best model on the training data

best_model.fit(X_train, y_train)


# Evaluate the best model
best_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
print("Best Model Accuracy:", best_accuracy)


# Predictions and SHAP values
y_pred = best_model.predict(X_test)

y_pred_prob = best_model.predict_proba(X_test)
print(metrics.classification_report(y_test, y_pred))

# Assuming y_test and y_pred are your test labels and predicted labels
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the report dictionary to a DataFrame
report_df = pd.DataFrame(report).transpose().reset_index()

#report_df.to_csv('./AlltypesAC_predict_accuracy_matrix.csv', index=False)

# Inverse transform to get original class names
original_classes = label_encoder_air.inverse_transform([0, 1, 2, 3])  # adjust if there are more or fewer classes
class_name_map = dict(zip([str(i) for i in range(len(original_classes))], original_classes))

# Get classification report as dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Replace numeric index with original class labels
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df['index'] = report_df['index'].replace(class_name_map)

# Optional: Clean formatting
report_df.rename(columns={'index': 'AC_Type'}, inplace=True)

# Display the final table
print(report_df)
report_df.to_csv('./AlltypesAC_predict_accuracy_matrix.csv', index=False)

#df_pred = pd.DataFrame(y_pred, columns=['pred'])
#see if this works
#df['yes_pred'] = y_pred


#accuracy matrix by urbanrural
urban_labels = df.loc[y_test.index, 'cityrural']

# Create a DataFrame to group by urbanicity
eval_df = pd.DataFrame({
    'true': label_encoder_air.inverse_transform(y_test),
    'pred': label_encoder_air.inverse_transform(y_pred),
    'urbanicity': urban_labels
})

# Calculate classification report by urbanicity
for group in ['Urban', 'Rural']:
    print(f"Classification Report for {group} Tracts:")
    subset = eval_df[eval_df['urbanicity'] == group]
    report = classification_report(
        subset['true'], 
        subset['pred'], 
        labels=label_encoder_air.classes_,
        output_dict=True,
        zero_division=0
    )
    # Convert to DataFrame for cleaner export or print
    report_df = pd.DataFrame(report).transpose()
    print(report_df.round(2))

report_df.to_csv('./AlltypesAC_urbanrural_predict_accuracy_matrix.csv')


    

# Compute confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Inverse transform the encoded labels to get original labels
y_test_original = label_encoder_air.inverse_transform(y_test)
y_pred_original = label_encoder_air.inverse_transform(y_pred)

# Get all possible class labels
all_labels = label_encoder_air.classes_

# Compute confusion matrix with all labels specified
cm = confusion_matrix(y_test_original, y_pred_original, labels=all_labels)

# Convert to DataFrame with labeled rows and columns
cm_df = pd.DataFrame(
    cm,
    index=[f"True: {label}" for label in all_labels],
    columns=[f"Pred: {label}" for label in all_labels]
)

# Save the confusion matrix as a CSV file
cm_df.to_csv('./confusion_matrix_allAC.csv', index=True)

# Plot and save the confusion matrix as a figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('./confusion_matrix_allAC.png', dpi=300)
plt.close()



'''


#visualization for model accuracy 
import pandas as pd

# Create a DataFrame from the trials object
results = pd.DataFrame(trials.results)
params = pd.DataFrame(trials.vals)

# Combine the parameters and results
params['loss'] = results['loss']
params['accuracy'] = -params['loss']

# Define the parameters you want to plot
params_to_plot = ['max_depth', 'n_estimators', 'gamma', 'reg_alpha', 'min_child_weight', 'colsample_bytree']

# Determine the number of rows/columns based on the number of parameters
n_cols = 2
n_rows = (len(params_to_plot) + 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows))
axes = axes.flatten()  # Flatten in case of only one row

for idx, param in enumerate(params_to_plot):
    # Create a pivot table where each entry is the mean accuracy for one combination of parameters
    pivot_table = params.pivot_table(
        values='accuracy',
        index=param,
        aggfunc=np.mean
    )

    # Sort the index if it is numeric
    if pd.api.types.is_numeric_dtype(pivot_table.index):
        pivot_table.sort_index(inplace=True)

    # Plotting
    pivot_table.plot(ax=axes[idx], title=f'Accuracy per {param}', legend=False, marker='o')
    axes[idx].set_xlabel(param)
    axes[idx].set_ylabel('Mean Accuracy')
    axes[idx].grid(True)

# Adjust the layout
plt.tight_layout()
plt.show()
plt.savefig('./AlltypesAC_predict_parameter_tuning.png', bbox_inches='tight')



'''



#origial data prediction

# Extract the features from your original data
X_original = DF[features]  # Replace 'target' with the actual target column name in your original data if it exists
y_original = DF[target]  # Replace 'target' with the actual target column name in your original data

# Use the best model to make predictions
original_predictions = best_model.predict(X_original)

# If you want to evaluate the predictions (assuming you have the true labels in 'y_original')
original_accuracy = accuracy_score(y_original, original_predictions)
print("Original Data Accuracy:", original_accuracy)

# Create a DataFrame to store the accuracy
accuracy_df = pd.DataFrame({'Original_Accuracy': [original_accuracy]})

# Save the DataFrame to a CSV file
#accuracy_df.to_csv('./alltype_original_accuracy_results.csv', index=False)


from sklearn.preprocessing import LabelEncoder

# Now, to reverse the encoding:
DF['final_prediction'] = original_predictions
DF['finalAIRCONDITIONINGpre'] = label_encoder_air.inverse_transform(DF['final_prediction'])

# Predict class probabilities
probs = best_model.predict_proba(X_original)


DF['probability'] = probs

# Property-level results with the predicted AC type (column "finalAIRCONDITIONINGpre").
# This file is the input of ACprob_to_county_zip.py.
DF.to_csv('./alltype_ac_predict_result.csv', index= False)


#creating underatintiy code

# Convert to DataFrame with class labels from the label encoder
class_labels = label_encoder_air.inverse_transform(np.arange(probs.shape[1]))
prob_df = pd.DataFrame(probs, columns=[f"prob_{label}" for label in class_labels])

# Predict most likely class
predicted_class_idx = np.argmax(probs, axis=1)
predicted_class_labels = label_encoder_air.inverse_transform(predicted_class_idx)

# Add predictions to test data
results = X_original.reset_index(drop=True).copy()
results["predicted_AC_type"] = predicted_class_labels

# Concatenate with probabilities
results = pd.concat([results, prob_df], axis=1)

# If you split DF into train and test, align results with original indices
results.index = X_original.index  # Reassign original indices

# Append predictions to original DF (only for test rows)
DF_with_preds = DF.copy()

# Drop existing prediction/probability columns if they already exist
cols_to_add = results.columns
existing_cols = DF_with_preds.columns.intersection(cols_to_add)
DF_with_preds.drop(columns=existing_cols, inplace=True)

# Add new prediction/probability columns only to test rows
DF_with_preds.loc[X_original.index, cols_to_add] = results

# Select only required columns before saving
prob_cols = [col for col in DF_with_preds.columns if col.startswith("prob_")]
final_cols = ['GEOID', 'GEOIDCN', 'PROPID', 'LAT', 'LON', 'COUNTY', 'STATE'] + prob_cols
DF_with_preds = DF_with_preds[final_cols]


# Property-level predicted probability of each AC type (used for the uncertainty layer)
DF_with_preds.to_csv("./prediction_probabilities.csv", index=False)



'''
# Here i am trying by state level 

# Assuming best_model is already defined and fitted
explainer = shap.Explainer(best_model)

#DF = df.loc[(df['STATE']== 'TX')& (df['COUNTY']=='Harris')|(df['COUNTY']=='Collin')]

# Make sure 'features' is defined as your list of model features
for fips_code, group in df.groupby('GEOIDCN'):
    X_county = group[features]
    
    # Calculate SHAP values for the county
    shap_values_county = explainer(X_county)
    
    # Use FIPS code as the file identifier, ensuring it's a string to handle any file naming issues
    file_identifier = str(fips_code)
    
    # Save SHAP values for the county
    with open(f'./YesAC_predict_shap_values_{file_identifier}.pkl', 'wb') as f:
        pickle.dump(shap_values_county, f)
    
    print(f'SHAP values calculated and saved for FIPS code {fips_code}')
    
    
    
    
    
# making the data for each county that i can use for making maps

import shap
import pickle
import pandas as pd

# Assuming best_model is already defined and fitted
explainer = shap.Explainer(best_model)

# Make sure 'features' is defined as your list of model features
result_dfs = []  # List to hold each group's result DataFrame

for fips_code, group in df.groupby('GEOIDCN'):
    X_county = group[features]
    
    # Calculate SHAP values for the county
    shap_values_county = explainer(X_county)
    print("Shape of SHAP values:", shap_values_county.shape)  # Debugging line to understand the shape

    # Handling SHAP values based on their shape
    if len(shap_values_county.shape) == 3:
        # Assuming we need the first set of SHAP values
        shap_values_county = shap_values_county[:, :, 0]

    # Convert SHAP values to a DataFrame for easier manipulation and concatenation
    shap_values_df = pd.DataFrame(shap_values_county.values, columns=features, index=X_county.index)
    
    # Make sure 'PROPID' is included in the shap_values_df to align with 'group'
    shap_values_df['PROPID'] = group['PROPID'].values  # Use .values to ensure correct alignment

    # Reset indices before concatenation to ensure alignment
    group_reset = group.reset_index(drop=True)
    shap_values_df_reset = shap_values_df.reset_index(drop=True)

    # Concatenate the SHAP values DataFrame with the original data group
    result_df = pd.concat([group_reset, shap_values_df_reset], axis=1)
    result_dfs.append(result_df)  # Append the result DataFrame to the list

    # Save each county's DataFrame as a CSV file, named by its GEOIDCN code
    filename = f"shap_values_county_{fips_code}.csv"
    result_df.to_csv(filename, index=False)
    print(f"Saved {filename}")

# Concatenate all result DataFrames from each group into a single DataFrame
final_result_df = pd.concat(result_dfs)

# Check the final DataFrame
print(final_result_df)

filename = "shap_values_county_level_all.csv"
final_result_df .to_csv(filename, index=False)



    

#test here if i can compile them
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
import glob
import os

# Use glob to find all pickle files in the results directory and subdirectories
file_paths = glob.glob('./results/**/*.pkl', recursive=True)
file_paths =file_paths[1:10]

# Function to load a pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Load SHAP Explanation objects
explanations = [pickle.load(open(file, 'rb')) for file in file_paths]

# Combine the SHAP values and base values
# Combine the SHAP values and data
combined_shap_values = np.vstack([e.values for e in explanations])
combined_data = np.vstack([e.data for e in explanations])
combined_data = np.vstack([e.data for e in explanations])
feature_names = [explanation.feature_names for explanation in explanations if hasattr(explanation, 'feature_names')][1]


# Update X_original with the correct data (make sure it matches combined_data)
X_original = pd.DataFrame(combined_data, columns=X_original.columns)

# Assuming X_original is loaded
# Rename columns as specified
# Rename columns as needed
feature_name_mapping = {
    'TOTROOMS': 'Total rooms',
    'CONDITION': 'Condition',
    'RENOYEAR': 'Renovated year',
    'HEAT': "Heater",
    'median_income': 'Median income',
    'prop_Black': '%Black/African Americans',
    'prop_All_Spanish': '%Hispanics',
    'prop_HigherEdu': "Post Secondary Education Rate",
    'CDDs': 'Cooling Degree Days',
    'HRI2020': 'Historical Housing Policy Score'
}
X_original_renamed = X_original.rename(columns=feature_name_mapping)


#shap.summary_plot(combined_shap_values, X_original_renamed, feature_names=X_original_renamed.columns, plot_type='dot', show=False)


# Generate SHAP summary plot
plt.figure(figsize=(10, 10))
shap.summary_plot(combined_shap_values, X_original_renamed, feature_names=X_original_renamed.columns, plot_type='dot', show=False)

# Add color bar to the plot
cb = plt.colorbar(label="Feature value (red high, blue low)")
cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)
cb.ax.yaxis.label.set_position((1.1, 0.5))

# Adjust subplot parameters and layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import shap

# Assuming combined_shap_values and X_original_renamed are already defined

# Extract SHAP values for a single output if your model is multi-output
# Adjust the index 0 if you want another class/output
shap_values_single_output = combined_shap_values[:, :, 0]

# Visualize the global feature importance using summary plot
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values_single_output, X_original_renamed, feature_names=X_original_renamed.columns.tolist(), show=False)

# Create color bar and set its label
#cb = plt.colorbar(label="Feature value (red high, blue low)")
#cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)

# Save the plot as PNG with high resolution
plt.savefig('./Local_Importance_summary_plot_with_colorbar.png', format='png', dpi=300, bbox_inches='tight')
plt.close()

'''




'''
# Calculate SHAP values
# Create SHAP Explainer

explainer = shap.Explainer(best_model, X_original)

# Calculate SHAP values for all instances
shap_values = explainer(X_original)



# Initialize the SHAP explainer
explainer = shap.Explainer(best_model, X_original)

# Calculate SHAP values for all instances
shap_values = explainer(X_original)

# Save the explainer and SHAP values using pickle
with open('./YesAC_predict_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

with open('./YesAC_predict_shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f)

# Load the explainer
#with open('./explainer.pkl', 'rb') as f:
#    explainer = pickle.load(f)

# Load the SHAP values
#with open('./shap_values.pkl', 'rb') as f:
#    shap_values = pickle.load(f)
    

#Global Importance
shap_values = explainer(X_original)

# Set larger figure size
plt.figure(figsize=(10, 10))  # You can adjust the width and height to fit your specific needs

# Define a custom dictionary for renaming columns
feature_name_mapping = {
    'TOTROOMS': 'Total rooms',
    'CONDITION': 'Condition',
    'RENOYEAR': 'Renovated year',
    'HEAT': "Heater",
    'median_income':'Median income',
    'prop_Black': '%Black/African Americans',
    'prop_All_Spanish': '%Hispanics',
    'prop_HigherEdu':"Post Secondary Education Rate",
    'CDDs':'Cooling Degree Days',
    'HRI2020':'Historical Housing Policy Score'  
    
}

X_original_renamed = X_original.rename(columns=feature_name_mapping)


# Visualize SHAP values

# Summary plot with a color bar
#shap.summary_plot(shap_values, X_original, feature_names=X_original.columns, plot_type='dot', show=False)
shap.summary_plot(shap_values, X_original_renamed, feature_names=X_original_renamed.columns, plot_type='dot', show=False)

# Add color bar to the plot
cb = plt.colorbar(label="Feature value (red high, blue low)")
cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)
cb.ax.yaxis.label.set_position((1.1, 0.5))

# Adjust subplot parameters and layout
plt.subplots_adjust(left=0.1, right=20, top=0.9, bottom=0.1, wspace=1, hspace=0.2)
plt.tight_layout()


plt.savefig('./Global_Importance_summary_plot_with_colorbar.png', dpi=300, bbox_inches='tight') # 

plt.show()


#need to test this out
shap.plots.heatmap(shap_values)


#local importance (This takes a long time)

# Initialize the SHAP explainer
explainer = shap.Explainer(best_model, X_original)

# Calculate SHAP values for all instances
shap_values = explainer(X_original)



    

feature_names = X_train.columns.tolist()  # Make sure this is defined before it's used
X_original_renamed = X_original.rename(columns=feature_name_mapping)


# Visualize the global feature importance using summary plot
plt.figure(figsize=(10, 10)) 
plt.tight_layout()
shap_values_single_output = shap_values.values[:, :, 0]  # Adjust the index 0 if you want another class/output
cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)
plt.figure()
shap.summary_plot(shap_values_single_output, X_original_renamed , feature_names=X_original_renamed.columns.tolist(), show=False)
plt.savefig( './Local_Importance_summary_plot_with_colorbar.png',  format='png', dpi=300, bbox_inches='tight')  # Save as SVG for higher quality
plt.close()




plt.close()  # Close the plot to free up memory

shap.dependence_plot(X_original_renamed.columns[1], shap_values.values[:, :, 0], X_original_renamed, xmin="percentile(1)", xmax="percentile(99)", interaction_index=None, show=False)
shap.dependence_plot(X_original_renamed.columns[1], shap_values.values[:, :, 0], X_original_renamed, interaction_index=X_original_renamed.columns[0],cmap=plt.get_cmap("cool"), xmin="percentile(1)", xmax="percentile(99)", show=False)
plt.savefig('./dependece_var1.png', dpi=300)  # Save as SVG for higher quality

shap.force_plot(explainer.expected_value,  shap_values[0, :])


# Geographical SHAP Visualization
# Assuming you have a GeoDataFrame with 'latitude' and 'longitude'
#geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
#geo_df['shap_value'] = shap_values.values[:, shap_values.feature_names.index('feature_name')]

'''
