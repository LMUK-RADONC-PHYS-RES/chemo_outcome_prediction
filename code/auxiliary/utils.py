import os
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import math
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts


def replace_nans(df):
    "Count and replace NaNs in pandas dataframe with medians."
    
    # Check for NaN values in the DataFrame
    has_nans = df.isna().any().any()

    nan_counts = 0
    if has_nans:
        # Count the number of NaNs in each column
        nan_counts = df.isna().sum()

        # Replace NaN values with the median of each column
        df = df.fillna(df.median())

    return df, nan_counts

# Define a custom function to apply the replacement logic
def replace_values(x):
    if x == '<10':
        return 0
    elif x == '>60':
        return 1
    elif  math.isnan(x):
        return x  # nans are adressed with other function
    elif int(x) > 60:
        return 1
    elif int(x) <= 60:
        return 0
    else:
        raise ValueError('Attention: unexpected value!')
    
def preprocess_data(dataframe, endpoint_status_header, endpoint_times_header, 
                    chemo_status_header='Chemotherapie?', 
                    timepoint=24, output_times=False, merge_hpv_u_and_n=True):
    """Drop unecessary columns and get binary labels for clinical endpoint.

    Args:
        dataframe (pandas dataframe): xtabular data read in pandas dataframe
        chemo_status_header (string): name of chemo status column
        endpoint_status_header (string): name of endpoint (eg PFS) status colum
        endpoint_times_header (string): name of endpoint (eg PFS) times column
        timepoint: timepoint for stratification in months
        output_times: if True, return also the times of the endpoint.
        merge_hpv_u_and_n: merge the unknow and negative HPV patient labeles to negative (0).

    Returns:
        cleaned dataframe and binary labels.
    """
    
    # List of columns to exclude
    include_columns = [
        'Alter bei Strahlentherapie',
        'Geschlecht\n(1: weiblich, 2: männlich)',
        'Lokalisation\n(1=Mundhöhle, 2=Oropharynx, 3=Hypopharynx, 4=Larynx, 5=Multilevel)',
        'T',
        'N',
        'Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)', # large nr unknown... introduced unknown class or merged with 0
        'Grading',
        'Nikotinanamnese,\nd.h. >10 pack years (0=nein, 1=ja)',
        'ECOG',
        'Age-adjusted\nCharlson Comorbidity Index',
        'CCI',
        'Hb (g/dl)\nzu Beginn (max. 5d vor Beginn)',
        'Leukozyten (tsd/µl)\nzu Beginn (max. 5d vor Beginn)',
        'CRP (mg/L)\nzu Beginn (max. 5d vor Beginn)',
        'Kreatinin (mg/dl)\nzu Beginn (max. 5d vor Beginn)',
        'GFR (ml/min*1.73m2 nach MDRD-Formel)\nzu Beginn (max. 5d vor Beginn)',  # problematic variables (>60 and nrs)...  fixed manually
        chemo_status_header, endpoint_status_header, endpoint_times_header
    ]

    # Drop cetrain columns
    dataframe = dataframe[include_columns]
    
    # Drop rows of patients where GFR is NaN
    dataframe = dataframe.dropna(subset=['GFR (ml/min*1.73m2 nach MDRD-Formel)\nzu Beginn (max. 5d vor Beginn)'])
    
    # Drop all rows for patients who did not receive chemo
    dataframe = dataframe[dataframe[chemo_status_header] == 1]
    dataframe = dataframe.drop(columns=chemo_status_header) # exclude column as it's not needed anymore
    
    # Drop all rows for patients who survived but follow-up was below timepoint months
    dataframe = dataframe[(dataframe[endpoint_status_header] == 1) | (dataframe[endpoint_times_header] >= timepoint)]
    
    # Get patients who survived/pfs more than timepoint months
    good = dataframe[dataframe[endpoint_times_header] >= timepoint]
    
    # Get patients who survived/pfs less than timepoint months
    bad = dataframe[(dataframe[endpoint_status_header] == 1) & (dataframe[endpoint_times_header] < timepoint)]

    # Print stats
    print(f'Good survivors with Chemo: {len(good)} patients / {len(good)/len(dataframe)*100} %')
    print(f'Bad survivors with Chemo: {len(bad)} patients / {len(bad)/len(dataframe)*100} % (PR baseline)') 
    
    # Generate a column with binary labels (good=0, bad=1)
    binary_labels = ((dataframe[endpoint_status_header] == 1) & (dataframe[endpoint_times_header] < timepoint)).astype(int)

    # get times for KM analysis
    times = dataframe[endpoint_times_header]

    # Drop survival columns as they are in the labels vector now
    dataframe = dataframe.drop(columns=[endpoint_status_header, endpoint_times_header])

    # For GFR categorize into <60 and >60 (not needed anymore as Alex converted into nrs)
    # dataframe['GFR (ml/min*1.73m2 nach MDRD-Formel)\nzu Beginn (max. 5d vor Beginn)'] = dataframe['GFR (ml/min*1.73m2 nach MDRD-Formel)\nzu Beginn (max. 5d vor Beginn)'].apply(replace_values)
    
    if merge_hpv_u_and_n:
        # For HPV status merge 'unknown' class with negative class (0)
        dataframe['Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)'] = dataframe['Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)'].fillna(0.0)      
    else:    
        # For HPV status introduce 'unknown' class (2)
        dataframe['Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)'] = dataframe['Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)'].fillna(2)
    print('Occurence of HPV-neg (0):', len(dataframe[dataframe["Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)"] == 0.0]))
    print('Occurence of HPV-pos (1):', len(dataframe[dataframe["Wenn Oropharynxkarzinom, HPV/p16-Status\n(0=negativ, 1=positiv)"] == 1.0]))
    
    # Replace all remaining NaNs with medians of each colum
    dataframe, nan_counts = replace_nans(dataframe)
    print(f'Following NaNs were replaced: \n{nan_counts}\n')
    
    # Scale numeric data to be roughly in the [0,1] range
    dataframe['Alter bei Strahlentherapie'] = dataframe['Alter bei Strahlentherapie']/100
    dataframe['Hb (g/dl)\nzu Beginn (max. 5d vor Beginn)'] = dataframe['Hb (g/dl)\nzu Beginn (max. 5d vor Beginn)']/10
    dataframe['Leukozyten (tsd/µl)\nzu Beginn (max. 5d vor Beginn)'] = dataframe['Leukozyten (tsd/µl)\nzu Beginn (max. 5d vor Beginn)']/10
    
    if output_times:
        return dataframe, binary_labels, times
    else:
        return dataframe, binary_labels


class ModelSavingCallback(xgb.callback.TrainingCallback):
    def __init__(self, watchlist, metrics, path_saving):
        """
        Initialize the ModelSavingCallback.

        :param watchlist: The watchlist parameter used in XGBoost training.
        :param metrics: The name of the metrics to monitor.
        :param path_saving: The path to save the model when the metric improves.
        """
        self.watchlist = watchlist
        self.metric_names = metrics
        self.path_saving = path_saving
        self.best_metric_sum = None

    def after_iteration(self, model, epoch, evals_log):
        """
        The callback function that is called after each iteration.

        :param model: The XGBoost model.
        :param epoch: The current iteration/epoch.
        :param evals_log: A dictionary containing evaluation results.
        """           
        current_metric_sum = 0.0
        # lopp over validation loss/metrics
        for metric_name in evals_log['val']:
            metric_value = evals_log['val'][metric_name][-1]  # get latest metric value
            if (metric_name in self.metric_names) and ('loss' not in metric_name):
                current_metric_sum += metric_value
            else:
                # for the losses take 1-value to be consistent in summation
                current_metric_sum += (1 - metric_value)
                    
        # Check if the sum of metrics improved
        if self.best_metric_sum is None or current_metric_sum > self.best_metric_sum:
            # Update the best metric sum and iteration
            self.best_metric_sum = current_metric_sum

            # Save the model
            model.save_model(os.path.join(self.path_saving, f'best_model_epoch{epoch:05d}_metric{current_metric_sum:.5f}.json'))

        # Return False to indicate that training should not stop
        return False


def get_path_to_best_model(path_model_files, model_format):
    # load trained model
    model_files = [] 
    metrics = [] 
    # loop over all subfolders and files of one pre training
    for _, _, file_list in os.walk(path_model_files):
        for file in file_list:
            if model_format == 'json':
                if file[-4:] == 'json':
                    model_files.append(file)
                    # append all the metrics values (attention: nr of digits is hard-coded)
                    metrics.append(float(file[-12:-5]))
            elif model_format == 'pth':
                if file[-3:] == 'pth':
                    model_files.append(file)
                    # append all the metrics values (attention: nr of digits is hard-coded)
                    metrics.append(float(file[-11:-4])) 
            else:
                raise ValueError('Attention: unknown format specified!')          
                    
    metrics = np.array(metrics)
    model_files = np.array(model_files)

    # find best model by looking at the largest metric
    best_model = model_files[np.argmax(metrics)]
    path_best_model = os.path.join(path_model_files, best_model) 
    
    return path_best_model


def compute_pvalue_and_plot_km(times, labels, high_risk_patient_indices, low_risk_patient_indices, endpoint, path):
    
    # compute p-value from log-rank test to check whether there is a significant ground truth difference between the two groups
    results = logrank_test(times[high_risk_patient_indices], times[low_risk_patient_indices], \
                            labels[high_risk_patient_indices], labels[low_risk_patient_indices]) 
    
    ax = plt.subplot(111)

    # Kaplan-Meier for Low-risk group
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(times[low_risk_patient_indices]/12, event_observed=labels[low_risk_patient_indices], label="Low-risk")
    kmf_low.plot(ax=ax, show_censors=True, ci_show=False, censor_styles={'marker': '+'})

    # Kaplan-Meier for High-risk group
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(times[high_risk_patient_indices]/12, event_observed=labels[high_risk_patient_indices], label="High-risk")
    kmf_high.plot(ax=ax, show_censors=True, ci_show=False, censor_styles={'marker': 'x'})

    # Add number at risk below the plot
    xticks = np.arange(0, 14, 2) # specify x-axis
    add_at_risk_counts(kmf_high, kmf_low, ax=ax, xticks=xticks, fontsize=16)
    
    # plt.subplots_adjust(left=0.2,bottom=0.3)  # Adjust left and right as needed

    ax.legend(loc='center left')
    ax.get_legend().remove()
    textstr = 'p-value = ' + str(round(results.p_value, 4))
        
    plt.text(0.23, 0.05, textstr, horizontalalignment='center', \
                 verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel('Time (years)')
    if endpoint == 'PFS':
        ax.set_ylabel('Progression-free survival')
    elif endpoint == 'OS':
        ax.set_ylabel('Overall survival')
    else:
        raise ValueError('Unknown endpoint!')
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim([0, 1.01])
    ax.legend(loc='upper right') 
    
    # set xlim steps
    ax.xaxis.set_ticks(xticks)
    
    # plt.tight_layout()
    plt.savefig(path + 'km.png', bbox_inches='tight')
    plt.show()
    plt.close() 
    
    
def plot_roc_auc(fpr, tpr, roc_auc, results_path, show=False):
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, 'b-', label = 'ROC-AUC = %0.2f' %roc_auc)
    plt.plot([0,1], [0,1], 'r--', label = 'No skill ROC-AUC = 0.5')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    #plt.title('Model Averaged ROC-AUC')
    
    # save figure 
    path_figure = os.path.join(results_path + 'ROC_AUC.png')
    plt.savefig(path_figure, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
def plot_pr_auc(labels, recall, precision, pr_auc, results_path, show=False):
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, marker='.', label='PR-AUC = %0.2f' %pr_auc)
    no_skill = len(labels[labels == 1])/len(labels)
    plt.plot([0,1], [no_skill, no_skill], 'r--', label='No skill PR-AUC = %0.2f' %no_skill)
    plt.legend(loc='lower left')
    plt.ylabel('Precision (PPV)')
    plt.xlabel('Recall (TPR)')
    plt.ylim(-0.05, 1.05) 
    #plt.title('Model Averaged PR-AUC')
    
    # save figure 
    path_figure = os.path.join(results_path + 'PR_AUC.png')
    plt.savefig(path_figure, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


