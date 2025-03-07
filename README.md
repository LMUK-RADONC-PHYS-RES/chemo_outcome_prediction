# Outcome prediction for elderly head&neck cancer patients receiving chemotherapy

Framework to train and validate an artficial neural network (ANN) for the prediction of progression-free-survival (PFS) or overall survival (OS).

Elia Lombardo and Sebastian Marschner

Department of Radiation Oncology, LMU University Hospital, LMU Munich, Germany

[Elia.Lombardo@med.uni-muenchen.de](mailto:Elia.Lombardo@med.uni-muenchen.de)
[Sebastian.Marschner@med.uni-muenchen.de](mailto:Sebastian.Marschner@med.uni-muenchen.de)

## Installation

* Download and unzip the repository to a local folder of your preference.
* Build a Docker image based on the provided `Dockerfile` and `requirements.txt` and run a container while mounting the `chemo_outcome_prediction` folder.
* Move your dataset excel file into the `data` subfolder.

## Usage

* Open `chemo_outcome_prediction/code/config.py` and change `path_project` to the path inside the container where `chemo_outcome_prediction` was mounted.
* Modify also other variables such as the excel filename of your data etc. as needed in `chemo_outcome_prediction/code/config.py`.
* Start a hyper-parameter grid search directly from the terminal by running for instance `bash main_grid_search.sh ANN OS` or `bash main_grid_search.sh ANN PFS` for training the ANN for PFS and OS, respectively.
* Perfom inference by setting the endpoint and model in the `main_infer_ANN.py` script. 
	* Trained model weights for OS and PFS can be found under `chemo_outcome_prediction/results/training/ANN`
	* SHAP explainability analysis included 

## Publication
If you use this code in a scientific publication, please cite our paper: 
https:xxx
