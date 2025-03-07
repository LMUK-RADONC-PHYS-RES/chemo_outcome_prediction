"""
Configuration file for main scripts
"""

#%%
import os
import matplotlib
import sys

#%%
# SET PATHS

# set path to project folder 
# (to be adapted to your project download location or
# to path inside docker container where project folder is mounted)
path_project = '/workspace/'

# path to data folder and to filename of input excel with covariates
path_project_data = os.path.join(path_project, 'data')
input_excel_file_train = 'data_train_val.xlsx'  # for training set
input_excel_file_test_1 = 'data_test.xlsx'  # for first testing set
input_excel_file_test_2 = 'data_test_2.xlsx'  # for second testing set

# path to model buiding code folder
path_project_code = os.path.join(path_project, 'code')

# path to results folder
path_project_results = os.path.join(path_project, 'results')

# add project auxiliary and models folder to Python path to be able to import self written modules from anywhere
sys.path.append(os.path.join(path_project_code, 'auxiliary'))

#%%
# SET OTHER SETTINGS 

# plotting  
matplotlib.rcParams.update({'font.size': 18})  # increase fontsize of all plots
plot = True # whether to plot results when running scripts

# neptune.ai can be used to track model trainings efficiently, account needed.
# neptune.ai api token and project name -> replace with your token!
os.environ['NEPTUNE_API_TOKEN'] = '...'    
neptune_project = "your_neptune_project"
