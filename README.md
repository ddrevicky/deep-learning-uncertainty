# Deep Learning Model Uncertainty in Medical Image Analysis

This repository contains the code for my Master's Thesis which deals with augmenting deep learning models with the ability to provide uncertainty estimates along with their predictions. I evaluate several uncertainty measures on a landmark localization task using a dataset of X-Ray cephalograms.

## Dataset
Follow the `prepare_dataset.ipynb` notebook to download and preprocess the data.

## Usage

Use the following scripts to evaluate the performance of the models on the landmark localization task.

##### Ensemble and Baseline
To train 15 independent Baseline models to form an Ensmeble as described in the thesis run `train_ensemble.sh`.
To generate predictions for all ensemble members and then evaluate the full Ensemble run `eval_ensemble.sh`.
Once this is done and the predictions are generated you can also evaluate a single Baseline member by running `eval_baseline.sh`.

##### MC-Dropout
Run `train_mc_dropout.sh` to train an MC-Dropout model as described in the thesis.
To first generate predictions on the test set using 15 samples and then evaluate the performance of the model run `eval_mc_dropout.sh`.
