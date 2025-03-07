# Outcome prediction for SENIOR trial patients receiving chemotherapy

Framework to train and validate two models for outcome prediction: an artficial neural network (ANN) and an extreme gradient boosting (XGB).

Elia Lombardo
LMU Munich
[Elia.Lombardo@med.uni-muenchen.de](mailto:Elia.Lombardo@med.uni-muenchen.de)

## Installation

* Download and unzip the repository to a local folder of your preference.
* Build a Docker image based on the provided `Dockerfile` and `requirements.txt` and run a container while mounting the `chemo_outcome_prediction` folder.
* Move the dataset file called `data_train_val.xlsx` into the `data` subfolder.

## Usage

* Open `chemo_outcome_prediction/code/config.py` and change `path_project` to the path inside the container where `chemo_outcome_prediction` was mounted.
* Start a hyper-parameter grid search directly from the terminal by running `bash main_grid_search.sh ANN` or `bash main_grid_search.sh XGB` for the ANN and XGB model, respectively. The second input argument can be used to specify the prediction endpoint, OS or PFS are available.
* SHAP explainability analysis included in the `main_infer` scripts.
