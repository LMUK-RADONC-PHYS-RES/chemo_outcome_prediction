#%%
import pandas as pd
import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Self written modules
import config
from auxiliary import utils, architectures

# Set path to best model and model parameters and dataset and endpoint
endpoint = 'OS'  # 'OS' or 'PFS'
if endpoint == 'OS':
	path_best_model = '/workspace/results/training/ANN/2024-12-17-16:58:07/best_model_epoch00047_metric1.87882.pth'
	params = {"hidden_size": 25} 
	path_saving = '/workspace/results/testing/ANN/2024-12-17-16:58:07_combined/'
elif endpoint == 'PFS':   
	path_best_model = '/workspace/results/training/ANN/2024-12-17-19:36:21/best_model_epoch00918_metric1.79990.pth'
	params = {"hidden_size": 20}
	path_saving = '/workspace/results/testing/ANN/2024-12-17-19:36:21_combined/'
else:
	raise ValueError('Unknown endpoint!')
	
path_to_km_threshold = os.path.join(os.path.dirname(path_best_model), 'median_pred.txt')

os.makedirs(path_saving, exist_ok=True)

# Load both datasets
df1 = pd.read_excel(os.path.join(config.path_project_data, config.input_excel_file_test_1))
df2 = pd.read_excel(os.path.join(config.path_project_data, config.input_excel_file_test_2))
if endpoint == 'PFS':
    df_clean_1, labels_1, times_1 = utils.preprocess_data(df1, 
                                                           endpoint_status_header='Status PFS\n(1=PFS-Event\n0=kein PFS-Event)', 
                                                           endpoint_times_header='Zeit PFS\n(in Monaten)', 
                                                           output_times=True)
    
    df_clean_2, labels_2, times_2 = utils.preprocess_data(df2, 
                                                           endpoint_status_header='Status PFS\n(1=PFS-Event\n0=kein PFS-Event)', 
                                                           endpoint_times_header='Zeit PFS\n(in Monaten)', 
                                                           output_times=True)
elif endpoint == 'OS':
    df_clean_1, labels_1, times_1 = utils.preprocess_data(df1, 
                                                           endpoint_status_header='Status OS \n(1=verstorben, 0=nicht verstorben)', 
                                                           endpoint_times_header='Zeit OS\n(in Monaten)', 
                                                           output_times=True)
    
    df_clean_2, labels_2, times_2 = utils.preprocess_data(df2, 
                                                           endpoint_status_header='Status OS \n(1=verstorben, 0=nicht verstorben)', 
                                                           endpoint_times_header='Zeit OS\n(in Monaten)', 
                                                           output_times=True)
else:
    raise ValueError('Unknown endpoint!')

# Combine the cleaned DataFrames
df_clean = pd.concat([df_clean_1, df_clean_2], ignore_index=True)
labels = pd.concat([labels_1,labels_2], ignore_index=True)
times = pd.concat([times_1,times_2], ignore_index=True)

# Show remaining columns
print('Remaining columns:')
print(df_clean.columns.tolist())

# Save the combined DataFrame
output_path = os.path.join('/workspace/results/testing', 'df_clean_combined.xlsx')
df_clean.to_excel(output_path, index=True)

#%%# Convert data to PyTorch tensors
x_test = torch.tensor(df_clean.values, dtype=torch.float32)
y_test = torch.tensor(labels.values, dtype=torch.float32)

# Initialize the model
input_size = len(df_clean.columns)
output_size = 1
model = architectures.ANN(input_size, params["hidden_size"], output_size, dropout_prob=0)

# Load best model
model.load_state_dict(torch.load(path_best_model))
model.eval()

# Make predictions on the val set
y_pred = model(x_test)[:,0]

# Convert to np arrays
y_pred = y_pred.detach().numpy()
y_test = y_test.detach().numpy()

# Print predictions and labels
# print(y_pred)
# print(y_test)

print('\n')
# Calculate loss and AUC scores
roc_auc = roc_auc_score(y_test, y_pred)
print(f"Testing ROC-AUC: {roc_auc}")
# using average precision as it is more appropiate than aucpr because
# no optimistic interpolation is performed (https://towardsdatascience.com/the-wrong-and-right-way-to-approximate-area-under-precision-recall-curve-auprc-8fd9ca409064)
pr_auc = average_precision_score(y_test, y_pred)  
print(f"Testing PR-AUC: {pr_auc}") 

# make plot for ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# print(f'Alternative computation for ROC-AUC: {auc(fpr, tpr)}')
utils.plot_roc_auc(fpr, tpr, roc_auc, path_saving, show=True)

# make plot for PR AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# print(f'Alternative computation for ROC-AUC: {auc(recall, precision)}')
utils.plot_pr_auc(y_test, recall, precision, pr_auc, path_saving, show=True)

# %%
# KM ANALYSIS

# load median score threshold from txt
with open(path_to_km_threshold, 'r') as file:
    threshold = float(file.read())
print(f'(Validation) threshold used for stratification: {threshold}')

# find high and low risk groups using the threshold
high_risk_patient_indices = np.where(y_pred > threshold) # 1 means sure death
low_risk_patient_indices = np.where(y_pred < threshold)
print('Number of high risk patients: ' + str(len(high_risk_patient_indices[0])))
print('Number of low risk patients: ' + str(len(low_risk_patient_indices[0])))

# get original values for the labels as otherwise KM will show no deaths after two years
indices_to_keep = times_1.index
# loop over all rows in df and keep indicised _to_keep
for row in df1.index:
    if row not in indices_to_keep:
        df1.drop(row, inplace=True)
if endpoint == 'OS':
    labels_original_1 = df1['Status OS \n(1=verstorben, 0=nicht verstorben)']
elif endpoint == 'PFS':
    labels_original_1 = df1['Status PFS\n(1=PFS-Event\n0=kein PFS-Event)']
else:   
    raise ValueError('Unknown endpoint!')
indices_to_keep = times_2.index
# loop over all rows in df and keep indicised _to_keep
for row in df2.index:
    if row not in indices_to_keep:
        df2.drop(row, inplace=True)    
if endpoint == 'OS':
    labels_original_2 = df2['Status OS \n(1=verstorben, 0=nicht verstorben)']
elif endpoint == 'PFS':
    labels_original_2 = df2['Status PFS\n(1=PFS-Event\n0=kein PFS-Event)']
else:   
    raise ValueError('Unknown endpoint!')
labels_original = pd.concat([labels_original_1, labels_original_2], ignore_index=True)

# for each of the two groups, plot true survival distribution (Kaplan-Meier estimator)
utils.compute_pvalue_and_plot_km(times.values, labels_original, 
                                 high_risk_patient_indices[0], low_risk_patient_indices[0],
                                endpoint=endpoint, path=path_saving)  

# write roacauc, prauc and number of patients to file
with open(os.path.join(path_saving, 'metrics.txt'), 'w') as file:
    file.write(f'ROC-AUC: {roc_auc}\n')
    file.write(f'PR-AUC: {pr_auc}\n')
    file.write(f'Number of high risk patients: {len(high_risk_patient_indices[0])}\n')
    file.write(f'Number of low risk patients: {len(low_risk_patient_indices[0])}\n')
    
# %%
# FEATURE IMPORTANCE

# Define short feature names for visualization
# short_feature_names = [f'f{i}' for i in range(len(df_clean.columns.values))]
short_feature_names = ['Age', 'Sex', 'Tumor site', 'T', 'N', 'HPV-status', 
                       'Grading', 'Smoking', 'ECOG', 'Age-adjusted CCI', 'CCI', 
                       'Hb', 'Leucocytes', 'CRP', 'Creatinine', 'GFR']

# Define a prediction function
def f(df):
    x = torch.tensor(df.values, dtype=torch.float32)
    outputs = model(x)[:,0].detach().cpu().numpy()
    return outputs

#%%

# Fits the explainer
# https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#
explainer = shap.Explainer(f, df_clean, feature_names=short_feature_names)
# Calculates the SHAP values - It takes some time
shap_values = explainer(df_clean.values)

#%%
plt.figure()
shap.plots.bar(shap_values, max_display=len(short_feature_names), show=False)
plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_feature_importance.png'), bbox_inches='tight')

plt.figure()
shap.plots.beeswarm(shap_values, max_display=len(short_feature_names), show=False)
plt.savefig(os.path.join(path_saving, 'shap_feature_beeswarm.png'), bbox_inches='tight')

plt.figure()
shap.plots.heatmap(shap_values, max_display=len(short_feature_names), show=False)
plt.savefig(os.path.join(path_saving, 'shap_feature_heatmap.png'), bbox_inches='tight')

#%%
# Get directly SHAP value
shap_values_only = explainer.shap_values(df_clean.values)

# plt.figure()
# fig = shap.summary_plot(shap_values_only, df_clean, plot_type="bar", 
#                         show=False, feature_names=short_feature_names, 
#                         show_values_in_legend=True)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
# plt.savefig(os.path.join(path_saving, 'shap_feature_importance.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('ECOG', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_ecog.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('GFR', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_gfr.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('HPV-status', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_hpv.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('T', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_t.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('N', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_n.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('CRP', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_crp.png'), bbox_inches='tight')


plt.figure()
fig = shap.dependence_plot('Tumor site', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_tumor_site.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('CCI', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_tumor_site.png'), bbox_inches='tight')

plt.figure()
fig = shap.dependence_plot('Age-adjusted CCI', shap_values_only, df_clean, 
                           feature_names=short_feature_names, interaction_index=None)
# plt.xlabel('mean(|SHAP value|) (average impact on model output)')
plt.savefig(os.path.join(path_saving, 'shap_partial_dependence_tumor_site.png'), bbox_inches='tight')
#%%
