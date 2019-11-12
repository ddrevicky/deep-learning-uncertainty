# Deep Learning Model Uncertainty in Medical Image Analysis

This repository contains the code for my Master's Thesis [(full text)](https://www.vutbr.cz/www_base/zav_prace_soubor_verejne.php?file_id=198231) which deals with augmenting deep learning models with the ability to provide uncertainty estimates along with their predictions. We evaluate several uncertainty measures on a landmark localization task using a dataset of X-ray cephalograms.

## Motivation
**Deep convolutional neural networks (CNNs)** achieve super-human results in image analysis but their outputs lack reliable information about the uncertainty of their predictions which prevents their wide-spread adaptation in medicine.

In this work we:
 - **Propose and train a CNN** for the task of automatic cephalometric **landmark localization** on skull X-rays. This is usually done by a dentist manually and is a time-consuming and tedious process.
 - **Design and implement uncertainty metrics** accompanying the trained models which are able to provide estimates of how certain the network is of its predictions (i.e., how much we should trust the predicted landmark positions).

## Solution: Landmark Localization
- CNN was trained on a dataset [1] of 200 X-ray cephalograms (annotated with 19 landmark positions). 
- Training done via **heatmap regression** (annotated landmark location is used to create a 2D heatmap with a Gaussian spike at that location).

## Solution: Uncertainty Measures
We proposed three models with corresponding uncertainty measures:

**1. Baseline** is a CNN based on U-Net [2]. It uses the value of the **maximum heatmap activation** as a measure of how uncertain the CNN is with its prediction (we assume lower values indicate higher uncertainty).
**2. Ensemble** is an ensemble model of 15 Baseline models based on the work of Lakshminarayanan et al. [3]. 
**3. MC-Dropout** is a CNN based on U-Net but additionally contains dropout layers which randomly remove some weights from the network both during training and evaluation (making the CNN **stochastic** so that multiple evaluations of the same image do not produce the exact same output). Based on the research by Gal et al. [4].

For both Ensemble and MC-Dropout models the **prediction mean** is used as the final predicted landmark position and the **prediction variance** (of the ensemble members and Monte Carlo samples respectively) as the uncertainty measure. We assume that as variance increases so does the models’ uncertainty.

## Experiment 1: Can Uncertainty Measures Detect Skull Rotation?


## Experiment 2: Can Uncertainty Measures Detect Deformed Data?
- Models achieve performance close to state-of-the-art on the studied landmark localization task.
- Uncertainty measures were able to reliably detect data unsuitable for automatic evaluation.

## Conclusion


## References
- [1] 2016, Wang et al.: A benchmark for comparison of dental radiography analysis algorithms
- [2] 2015, Ronneberger, O.; Fischer, P.; Brox, T.: U-Net: Convolutional Networks forBiomedical Image Segmentation
- [3] 2017, Lakshminarayanan et al.: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- [4] 2016, Gal, Y.; Ghahramani, Z.: Dropout As a Bayesian Approximation 

## Usage

#### Preprocessing the dataset
Follow the `prepare_dataset.ipynb` notebook to download and preprocess the data.

#### Model evaluation

Use the following scripts to evaluate the performance of the models on the landmark localization task.

##### Ensemble and Baseline
To train 15 independent Baseline models to form an Ensmeble as described in the thesis run `train_ensemble.sh`.
To generate predictions for all ensemble members and then evaluate the full Ensemble run `eval_ensemble.sh`.
Once this is done and the predictions are generated you can also evaluate a single Baseline member by running `eval_baseline.sh`.

##### MC-Dropout
Run `train_mc_dropout.sh` to train an MC-Dropout model as described in the thesis.
To first generate predictions on the test set using 15 samples and then evaluate the performance of the model run `eval_mc_dropout.sh`.
