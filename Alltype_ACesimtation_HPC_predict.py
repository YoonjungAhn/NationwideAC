# !/usr/bin/env Python3
# -*- Coding: utf-8 -*-
"""
Created on Tue Jul 16 15:20:32 2024

@author: y943a214
"""
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

os.chdir("<your_path_here>")
# Os.chdir("<your_path_here>")


predictdf =pd.read_csv("<your_path_here>")
predictdf = pd.read_csv("<your_path_here>")
predictdf =predictdf [['PROPID','Yespredicted']]

df = pd.read_csv('./allcombined_for_contemporal_prediction.csv',  na_values=[''])
df = df.loc[~df['STATE'].isin(['AK', 'HI', 'PR'])]
df['YEARBUILT'] = round(df['YEARBUILT'])
df['TOTROOMS'] = round(df['TOTROOMS'])



df = pd.merge(df,predictdf, how='outer', on = ['PROPID'])

len(df.loc[df.AIRCONDITIONINGrecode=='NoAC'])
len(df.loc[df.AIRCONDITIONINGrecode=='nan'])


# Replace 'YES' Or NaN values in 'AIRCONDITIONINGrecode' with corresponding values from 'Yespredicted'
df.loc[df['AIRCONDITIONINGrecode'] == 'nan', 'AIRCONDITIONINGrecode'] = np.nan


df['AIRCONDITIONINGcombined'] = np.where(df['AIRCONDITIONINGrecode'].isin(['Yes']),
                                       df['Yespredicted'],
                                       df['AIRCONDITIONINGrecode'])

df.loc[df['AIRCONDITIONINGcombined'] == 'nan', 'AIRCONDITIONINGcombined'] = np.nan


# Predicting YES AC
# Df = pd.read_csv("<your_path_here>",  na_values=[''])
df['AIRCONDITIONINGrecode'].unique()
df['REHEATTYPE'].unique()
# DF = Df.copy()


# Encording the data
label_encoder_air = LabelEncoder()
label_encoder_heat = LabelEncoder()

df['AIRCONDITIONINGrecode_encoded'] = label_encoder_air.fit_transform(df['AIRCONDITIONINGcombined'].astype(str))
df['HEAT_encoded'] = label_encoder_heat.fit_transform(df['REHEATTYPE'].astype(str))
df['CDDs_quintiles'] = pd.qcut(df['CDDs'], 5, labels=[1, 2, 3, 4, 5])
df['CDDs_quintiles'].astype(str)
df['CONDITION'].astype(str)
df['HEAT_encoded'].astype(str)

# Define The feature columns and target column
features = df[['RENOYEAR', 'TOTROOMS','CONDITION','HEAT_encoded', 'BUPR','median_income','prop_Black','prop_All_Spanish','prop_HigherEdu','CDDs','HRI2020']].columns# CDDs Adjust indexing if necessary
target = 'AIRCONDITIONINGrecode_encoded'


# Create Stratified samples
# First, Create a column for stratification
df['stratify_col'] = df['CDDs_quintiles'].astype(str) + "_" + df['AIRCONDITIONINGrecode']
df['stratify_col'] = df['stratify_col'].fillna('Missing')
# Split The data into train and test data

# Perform Stratified split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['stratify_col'])


# Separate Features and target
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]



# Tuneing The model 
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
                    colsample_bytree=int(space['colsample_bytree']),enable_categorical=True, n_jobs=multiprocessing.cpu_count())

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
                        max_evals = 10,
                        trials = trials)

params = pd.DataFrame(trials.vals)


params.to_csv('./alltypeAC_model_training_trial_results.csv')


print("The best hyperparameters are: ", "\n")
print(best_hyperparams)

# Define The best model with the best hyperparameters
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


# Hyperparameter Value ranges
colsample_bytree_values = np.arange(0.6, 1.05, 0.05).tolist()
max_depth_range = list(range(5, 15))
n_estimators_range = list(range(200, 1001, 100))
min_child_weight_range = list(range(1, 10))


# Translate Indices to actual values
best_hyperparams['colsample_bytree'] = colsample_bytree_values[best_hyperparams['colsample_bytree']]
best_hyperparams['max_depth'] = max_depth_range[best_hyperparams['max_depth']]
best_hyperparams['n_estimators'] = n_estimators_range[best_hyperparams['n_estimators']]
best_hyperparams['min_child_weight'] = min_child_weight_range[best_hyperparams['min_child_weight']]

# Setup And train the model
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


best_model = xgb.XGBClassifier(
    **best_hyperparams,  # Unpack hyperparameters directly into function arguments
    enable_categorical=True,
    objective='multi:softprob',
    random_state=42,
    n_jobs=multiprocessing.cpu_count()  # Automatically use all available cores
)



# Fit The best model on the training data

best_model.fit(X_train, y_train)

# Evaluate The best model
best_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
print("Best Model Accuracy:", best_accuracy)


# Predictions And SHAP values
y_pred = best_model.predict(X_test)

y_pred_prob = best_model.predict_proba(X_test)
print(metrics.classification_report(y_test, y_pred))

# Assuming Y_test and y_pred are your test labels and predicted labels
report = classification_report(y_test, y_pred, output_dict=True)

# Convert The report dictionary to a DataFrame
report_df = pd.DataFrame(report).transpose().reset_index()

# Report_df.to_csv('./AlltypesAC_predict_accuracy_matrix.csv', index=False)


# Df_pred = pd.DataFrame(y_pred, columns=['pred'])
# See if this works
# Df['yes_pred'] = y_pred



# Visualization for model accuracy 
import pandas as pd

# Create A DataFrame from the trials object
results = pd.DataFrame(trials.results)
params = pd.DataFrame(trials.vals)

# Combine The parameters and results
params['loss'] = results['loss']
params['accuracy'] = -params['loss']

# Define The parameters you want to plot
params_to_plot = ['max_depth', 'n_estimators', 'gamma', 'reg_alpha', 'min_child_weight', 'colsample_bytree']

# Determine The number of rows/columns based on the number of parameters
n_cols = 2
n_rows = (len(params_to_plot) + 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5 * n_rows))
axes = axes.flatten()  # Flatten in case of only one row

for idx, param in enumerate(params_to_plot):
    # Create A pivot table where each entry is the mean accuracy for one combination of parameters
    pivot_table = params.pivot_table(
        values='accuracy',
        index=param,
        aggfunc=np.mean
    )

    # Sort The index if it is numeric
    if pd.api.types.is_numeric_dtype(pivot_table.index):
        pivot_table.sort_index(inplace=True)

    # Plotting
    pivot_table.plot(ax=axes[idx], title=f'Accuracy per {param}', legend=False, marker='o')
    axes[idx].set_xlabel(param)
    axes[idx].set_ylabel('Mean Accuracy')
    axes[idx].grid(True)

# Adjust The layout
plt.tight_layout()
plt.show()
plt.savefig('./AlltypesAC_predict_parameter_tuning.png', bbox_inches='tight')


# Origial data prediction

# Extract The features from your original data
X_original = df[features]  # Replace 'target' with the actual target column name in your original data if it exists
y_original = df[target]  # Replace 'target' with the actual target column name in your original data

# Use The best model to make predictions
original_predictions = best_model.predict(X_original)


# If You want to evaluate the predictions (assuming you have the true labels in 'y_original')
original_accuracy = accuracy_score(y_original, original_predictions)
print("Original Data Accuracy:", original_accuracy)

# Create A DataFrame to store the accuracy
accuracy_df = pd.DataFrame({'Original_Accuracy': [original_accuracy]})

# Save The DataFrame to a CSV file
# Accuracy_df.to_csv('./alltype_original_accuracy_results.csv', index=False)


from sklearn.preprocessing import LabelEncoder

# Now, To reverse the encoding:
df['final_prediction'] = original_predictions
df['finalAIRCONDITIONINGpre'] = label_encoder_air.inverse_transform(df['final_prediction'])


df.to_csv("<your_path_here>", index= False)



# Here I am trying by state level 

# Assuming Best_model is already defined and fitted
explainer = shap.Explainer(best_model)

# DF = Df.loc[(df['STATE']== 'TX')& (df['COUNTY']=='Harris')|(df['COUNTY']=='Collin')]

# Make Sure 'features' is defined as your list of model features
for fips_code, group in df.groupby('GEOIDCN'):
    X_county = group[features]

    # Calculate SHAP Values for the county
    shap_values_county = explainer(X_county)

    # Use FIPS Code as the file identifier, ensuring it's a string to handle any file naming issues
    file_identifier = str(fips_code)

    # Save SHAP Values for the county
    with open(f'./YesAC_predict_shap_values_{file_identifier}.pkl', 'wb') as f:
        pickle.dump(shap_values_county, f)

    print(f'SHAP values calculated and saved for FIPS code {fips_code}')





# Making the data for each county that i can use for making maps

import shap
import pickle
import pandas as pd

# Assuming Best_model is already defined and fitted
explainer = shap.Explainer(best_model)

# Make Sure 'features' is defined as your list of model features
result_dfs = []  # List to hold each group's result DataFrame

for fips_code, group in df.groupby('GEOIDCN'):
    X_county = group[features]

    # Calculate SHAP Values for the county
    shap_values_county = explainer(X_county)
    print("Shape of SHAP values:", shap_values_county.shape)  # Debugging line to understand the shape

    # Handling SHAP Values based on their shape
    if len(shap_values_county.shape) == 3:
        # Assuming We need the first set of SHAP values
        shap_values_county = shap_values_county[:, :, 0]

    # Convert SHAP Values to a DataFrame for easier manipulation and concatenation
    shap_values_df = pd.DataFrame(shap_values_county.values, columns=features, index=X_county.index)

    # Make Sure 'PROPID' is included in the shap_values_df to align with 'group'
    shap_values_df['PROPID'] = group['PROPID'].values  # Use .values to ensure correct alignment

    # Reset Indices before concatenation to ensure alignment
    group_reset = group.reset_index(drop=True)
    shap_values_df_reset = shap_values_df.reset_index(drop=True)

    # Concatenate The SHAP values DataFrame with the original data group
    result_df = pd.concat([group_reset, shap_values_df_reset], axis=1)
    result_dfs.append(result_df)  # Append the result DataFrame to the list

    # Save Each county's DataFrame as a CSV file, named by its GEOIDCN code
    filename = f"shap_values_county_{fips_code}.csv"
    result_df.to_csv(filename, index=False)
    print(f"Saved {filename}")

# Concatenate All result DataFrames from each group into a single DataFrame
final_result_df = pd.concat(result_dfs)

# Check The final DataFrame
print(final_result_df)

filename = "shap_values_county_level_all.csv"
final_result_df .to_csv(filename, index=False)




import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
import glob
import os

# Use Glob to find all pickle files in the results directory and subdirectories
file_paths = glob.glob('./results/**/*.pkl', recursive=True)
file_paths =file_paths[1:10]

# Function To load a pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Load SHAP Explanation Objects
explanations = [pickle.load(open(file, 'rb')) for file in file_paths]

# Combine The SHAP values and base values
# Combine The SHAP values and data
combined_shap_values = np.vstack([e.values for e in explanations])
combined_data = np.vstack([e.data for e in explanations])
combined_data = np.vstack([e.data for e in explanations])
feature_names = [explanation.feature_names for explanation in explanations if hasattr(explanation, 'feature_names')][1]


# Update X_original With the correct data (make sure it matches combined_data)
X_original = pd.DataFrame(combined_data, columns=X_original.columns)

# Assuming X_original Is loaded
# Rename Columns as specified
# Rename Columns as needed
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


# Shap.summary_plot(combined_shap_values, X_original_renamed, feature_names=X_original_renamed.columns, plot_type='dot', show=False)


# Generate SHAP Summary plot
plt.figure(figsize=(10, 10))
shap.summary_plot(combined_shap_values, X_original_renamed, feature_names=X_original_renamed.columns, plot_type='dot', show=False)

# Add Color bar to the plot
cb = plt.colorbar(label="Feature value (red high, blue low)")
cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)
cb.ax.yaxis.label.set_position((1.1, 0.5))

# Adjust Subplot parameters and layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import shap

# Assuming Combined_shap_values and X_original_renamed are already defined

# Extract SHAP Values for a single output if your model is multi-output
# Adjust The index 0 if you want another class/output
shap_values_single_output = combined_shap_values[:, :, 0]

# Visualize The global feature importance using summary plot
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values_single_output, X_original_renamed, feature_names=X_original_renamed.columns.tolist(), show=False)

# Create Color bar and set its label
# Cb = plt.colorbar(label="Feature value (red high, blue low)")
# Cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)

# Save The plot as PNG with high resolution
plt.savefig('./Local_Importance_summary_plot_with_colorbar.png', format='png', dpi=300, bbox_inches='tight')
plt.close()






# Calculate SHAP Values
# Create SHAP Explainer

explainer = shap.Explainer(best_model, X_original)

# Calculate SHAP Values for all instances
shap_values = explainer(X_original)



# Initialize The SHAP explainer
explainer = shap.Explainer(best_model, X_original)

# Calculate SHAP Values for all instances
shap_values = explainer(X_original)

# Save The explainer and SHAP values using pickle
with open('./YesAC_predict_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

with open('./YesAC_predict_shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f)

# Load The explainer
# With open('./explainer.pkl', 'rb') as f:
# Explainer = pickle.load(f)

# Load The SHAP values
# With open('./shap_values.pkl', 'rb') as f:
# Shap_values = pickle.load(f)


# Global Importance
shap_values = explainer(X_original)

# Set Larger figure size
plt.figure(figsize=(10, 10))  # You can adjust the width and height to fit your specific needs

# Define A custom dictionary for renaming columns
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


# Visualize SHAP Values

# Summary Plot with a color bar
# Shap.summary_plot(shap_values, X_original, feature_names=X_original.columns, plot_type='dot', show=False)
shap.summary_plot(shap_values, X_original_renamed, feature_names=X_original_renamed.columns, plot_type='dot', show=False)

# Add Color bar to the plot
cb = plt.colorbar(label="Feature value (red high, blue low)")
cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)
cb.ax.yaxis.label.set_position((1.1, 0.5))

# Adjust Subplot parameters and layout
plt.subplots_adjust(left=0.1, right=20, top=0.9, bottom=0.1, wspace=1, hspace=0.2)
plt.tight_layout()


plt.savefig('./Global_Importance_summary_plot_with_colorbar.png', dpi=300, bbox_inches='tight') #

plt.show()


# Need to test this out
shap.plots.heatmap(shap_values)


# Local importance (This takes a long time)

# Initialize The SHAP explainer
explainer = shap.Explainer(best_model, X_original)

# Calculate SHAP Values for all instances
shap_values = explainer(X_original)



feature_names = X_train.columns.tolist()  # Make sure this is defined before it's used
X_original_renamed = X_original.rename(columns=feature_name_mapping)


# Visualize The global feature importance using summary plot
plt.figure(figsize=(10, 10))
plt.tight_layout()
shap_values_single_output = shap_values.values[:, :, 0]  # Adjust the index 0 if you want another class/output
cb.set_label("Feature value (red high, blue low)", rotation=270, labelpad=20)
plt.figure()
shap.summary_plot(shap_values_single_output, X_original_renamed , feature_names=X_original_renamed.columns.tolist(), show=False)
plt.savefig( './Local_Importance_summary_plot_with_colorbar.png',  format='png', dpi=300, bbox_inches='tight')  # Save as SVG for higher quality
plt.close()


shap.dependence_plot(X_original_renamed.columns[1], shap_values.values[:, :, 0], X_original_renamed, xmin="percentile(1)", xmax="percentile(99)", interaction_index=None, show=False)
shap.dependence_plot(X_original_renamed.columns[1], shap_values.values[:, :, 0], X_original_renamed, interaction_index=X_original_renamed.columns[0],cmap=plt.get_cmap("cool"), xmin="percentile(1)", xmax="percentile(99)", show=False)
plt.savefig('./dependece_var1.png', dpi=300)  # Save as SVG for higher quality

shap.force_plot(explainer.expected_value,  shap_values[0, :])


