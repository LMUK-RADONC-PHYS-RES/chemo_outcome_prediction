#%%
import pandas as pd
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# Self written modules
import config
from auxiliary import utils, architectures

import neptune
run = neptune.init_run(
    project=config.neptune_project,
    tags=["ANN"],
) 

import argparse
# command line parsing instantiation
parser = argparse.ArgumentParser()

# command line options
parser.add_argument("--learning_rate", type=float, default=0.0001,
                    help="learning rate of optimizer")
parser.add_argument("--weight_decay", type=float, default=0.000001,
                    help="L2 regularization term on weights")
parser.add_argument("--dropout_prob", type=float, default=0.25,
                    help="droput regularization probability")
parser.add_argument("--hidden_size", type=int, default=10,
                    help="number of neurons in hidden layers")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size during learning")
parser.add_argument("--endpoint", type=str, default='PFS',
                    help="endpoint of prediction (PFS or OS)")
args = parser.parse_args()
#%%
# Time at which run is started, used to save results
start_time_string = time.strftime("%Y-%m-%d-%H:%M:%S")  
path_saving = os.path.join(config.path_project_results, 'training', 'ANN', start_time_string)
os.makedirs(path_saving, exist_ok=True)

# Read the Excel file into a Pandas DataFrame
file_path = os.path.join(config.path_project_data, config.input_excel_file_train)
df = pd.read_excel(file_path)

# Display the first few rows/cols of the DataFrame
# print(df.head())
#%%

# Drop unwanted patients and get binary labels (good=0, bad=1)
print(args.endpoint)
run["endpoint"] = args.endpoint
if args.endpoint == 'PFS':
    df_clean, labels = utils.preprocess_data(df, endpoint_status_header='Status PFS\n(1=PFS-Event\n0=kein PFS-Event)', endpoint_times_header='Zeit PFS\n(in Monaten)')
elif args.endpoint == 'OS':
    df_clean, labels = utils.preprocess_data(df, endpoint_status_header='Status OS \n(1=verstorben, 0=nicht verstorben)', endpoint_times_header='Zeit OS\n(in Monaten)')
else:
    raise ValueError('Unknown endpoint!')
print('Remaining columns:')
print(df_clean.columns.tolist()) # show all remaining columns

#%%
# Define hyperparameters
input_size = len(df_clean.columns)
output_size = 1
params = {"hidden_size": args.hidden_size, "learning_rate": args.learning_rate, 
          "dropout_prob": args.dropout_prob, "batch_size": args.batch_size, "weight_decay": args.weight_decay,
          "num_epochs": 1000}
run["parameters"] = params

# Split the data into training and validation 
x_train, x_val, y_train, y_val = train_test_split(df_clean, labels, test_size=0.20, random_state=42)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
x_val = torch.tensor(x_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)

# Create PyTorch DataLoader for training and validation sets
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)


# Initialize the model, loss function, and optimizer
model = architectures.ANN(input_size, params["hidden_size"], output_size, dropout_prob=params["dropout_prob"])
loss_criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

best_metric = 0
best_epoch = 0
early_stopping_patience = params["num_epochs"]/5
# Training loop
for epoch in range(params["num_epochs"]):
    print(f'----- Epoch [{epoch + 1}/{params["num_epochs"]}] -----')
    
    model.train()
    for inputs, labels in train_loader:
        # Forward pass (drop additional dim)
        outputs = model(inputs)[:,0]
        train_loss = loss_criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    run["train/loss"].append(train_loss)

    model.eval()
    val_loss = 0.0
    val_roc_auc = 0.0
    val_pr_auc = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Make predictions on the validation set and calculate metrics
            outputs = model(inputs)[:,0]
            val_loss += loss_criterion(outputs, labels).item()
            val_roc_auc += roc_auc_score(labels, outputs)
            val_pr_auc += average_precision_score(labels, outputs)

    val_loss /= len(val_loader)
    val_roc_auc /= len(val_loader)
    val_pr_auc /= len(val_loader)

    # Log metrics to Neptune
    run["val/loss"].append(val_loss)
    run["val/roc_auc"].append(val_roc_auc)
    run["val/pr_auc"].append(val_pr_auc)

    current_metric_sum = (1-val_loss) + val_roc_auc + val_pr_auc
    # Check if combined metric improved
    if current_metric_sum > best_metric:
        best_metric = current_metric_sum
        torch.save(model.state_dict(), os.path.join(path_saving, f'best_model_epoch{epoch:05d}_metric{current_metric_sum:.5f}.pth'))
        best_epoch = epoch
        
    # stop the optimization if the loss didn't decrease after early_stopping nr of epochs
    if early_stopping_patience is not None:
        if (epoch - best_epoch) > early_stopping_patience:
            print('Early stopping the optimization!')
            break

    
# Load best model
path_best_model = utils.get_path_to_best_model(path_saving, 'pth')
model.load_state_dict(torch.load(path_best_model))
model.eval()

# Upload best model to neptune
run[f"model/{os.path.basename(path_best_model)}"].upload(path_best_model)

# Make predictions on the val set
y_pred = model(x_val)[:,0]

# Convert to np arrays
y_pred = y_pred.detach().numpy()
y_val = y_val.detach().numpy()

# Print predictions and labels
print(y_pred)
print(y_val)
print(f'Median of predictions: {np.median(y_pred)}')
run["median_pred"] = np.median(y_pred)
print(f'Mean of predictions: {np.mean(y_pred)}')
run["mean_pred"] = np.mean(y_pred)

# Calculate loss and AUC scores
log_loss_value = log_loss(y_val, y_pred)
print(f"Validation log loss: {log_loss_value}")
roc_auc = roc_auc_score(y_val, y_pred)
print(f"Validation ROC-AUC: {roc_auc}")
# using average precision as it is more appropiate than aucpr because
# no optimistic interpolation is performed (https://towardsdatascience.com/the-wrong-and-right-way-to-approximate-area-under-precision-recall-curve-auprc-8fd9ca409064)
pr_auc = average_precision_score(y_val, y_pred)  
print(f"Validation PR-AUC: {pr_auc}") 

# Record numbers or text
run["val_log_loss"] = log_loss_value
run["val_roc_auc"] = roc_auc
run["val_pr_auc"] = pr_auc
run.stop()
