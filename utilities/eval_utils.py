import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import PIL
import shutil
from pathlib import Path
import time
import random

from .common_utils import *

def get_true_landmarks(annotations_path, image_path):
    ''' Returns an array of true landmarks for an image
    '''
    image_id = image_path.stem
    annots = (annotations_path/f'{image_id}.txt').read_text()
    annots = annots.split('\n')[:N_LANDMARKS]
    annots = [l.split(',') for l in annots]
    true_landmarks = [np.array([float(l[1]), float(l[0])]) for l in annots]  # Swap XY to YX order
    return np.array(true_landmarks)

FILE_COL = 'file'

def read_prediction_files_as_df(prediction_files):
    ''' Reads individual prediction files as dataframes and then concatenates them into a single dataframe with all predictions for all images and samples.
    '''
    dfs = []
    for f in prediction_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(FILE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_predictions_for_image(df, image_file, n_samples):
    ''' Extracts all of the landmark position and activations samples for the given image from the dataframe
        Returns them as numpy arrays.
    '''
    image_df = df.loc[df[FILE_COL] == image_file]
    image_df.reset_index(drop=True, inplace=True)
    computed_samples = image_df.shape[0]
    
    n_samples = min(n_samples, computed_samples)
    activation_samples = np.zeros((n_samples, N_LANDMARKS))
    landmark_samples = np.zeros((n_samples, N_LANDMARKS, 2))
    
    for i, row in image_df.iterrows():
        if i < n_samples:
            for lm in range(N_LANDMARKS):
                activation_samples[i, lm] = row[f'{lm}_act']
                landmark_samples[i, lm] = np.array([row[f'{lm}_y'], row[f'{lm}_x']])
    return landmark_samples, activation_samples


def get_landmark_prediction_variance(lm_samples):
    n_samples, n_landmarks, _ = lm_samples.shape
    landmark_mean = np.mean(lm_samples, axis=0)
    distances = np.zeros((n_samples, n_landmarks))
    for s in range(n_samples):
        for lm in range(n_landmarks):
            dist = np.linalg.norm(lm_samples[s, lm] - landmark_mean[lm])
            distances[s, lm] = dist
    return np.mean(distances, axis=0)


def get_predicted_landmarks_for_image(landmark_samples):
    ''' Returns the average predicted landmark positions in term of samples and their variance computed
        as the mean distance between landmarks and the mean landmark.
        Takes an array of dimension (n_samples, n_landmarks, 2).
    '''
    landmark_mean = np.mean(landmark_samples, axis=0)
    landmark_var = get_landmark_prediction_variance(landmark_samples)
    return landmark_mean, landmark_var/PIXELS_PER_MM


def get_predicted_activations_for_image(activation_samples):
    ''' Returns the average activation landmark positions and their variance in term of samples.
        Takes an array of dimension (n_samples, n_landmarks).
    '''
    activation_mean = np.mean(activation_samples, axis=0)
    activation_var = np.var(activation_samples, axis=0)
    return activation_mean, activation_var


def radial_error_mm(true, pred):
    ''' Returns the radial error in mms for a single landmark.
    '''
    return np.linalg.norm(pred/PIXELS_PER_MM - true/PIXELS_PER_MM)


def get_radial_errors_mm_for_image(true_landmarks, predicted_landmarks):
    ''' Returns an array containing the radial error for each landmark for the image.
    '''
    radial_errors = np.zeros(N_LANDMARKS)
    for lm in range(N_LANDMARKS):
        radial_errors[lm] = radial_error_mm(true_landmarks[lm], predicted_landmarks[lm])
    return radial_errors


def get_accuracy_metrics(radial_errors_mm_all):
    ''' Computes accuracy metrics from radial errors.
    '''
    mre = radial_errors_mm_all.mean()
    std = radial_errors_mm_all.std()
    
    sdr_2 = (radial_errors_mm_all < 2.0).mean()
    sdr_2_5 = (radial_errors_mm_all < 2.5).mean()
    sdr_3 = (radial_errors_mm_all < 3.0).mean()
    sdr_4 = (radial_errors_mm_all < 4.0).mean()
    
    return {'mre': mre, 'std':std,
            'sdr_2': sdr_2, 'sdr_2_5': sdr_2_5,
            'sdr_3': sdr_3, 'sdr_4': sdr_4}


def print_accuracy_metrics(m):
    print(f"MRE: {m['mre']:{4}.{4}} mm, STD: {m['std']:{4}.{4}} mm\
           \nSDR 2mm: {m['sdr_2']:{4}.{4}}\
           \nSDR 2.5mm: {m['sdr_2_5']:{4}.{4}}\
           \nSDR 3mm: {m['sdr_3']:{4}.{4}}\
           \nSDR 4mm: {m['sdr_4']:{4}.{4}}")   

          
def log_metrics(metrics, metrics_dir, n_samples):
    ''' Logs the computed metrics to a csv file in the metrics subdirectory of the log directory for the model.
    '''
    metrics['samples'] = n_samples
    metrics_df = pd.DataFrame(data=metrics, index=[0])
    metrics_df.to_csv(metrics_dir/f'{n_samples}.csv')
          
          
def get_test_predictions_df(model_log_dir):
    """ Load computed model predictions
    """
    prediction_dir = model_log_dir/'predictions'
    prediction_files = list_files(prediction_dir)
    predictions_df = read_prediction_files_as_df(prediction_files)
    return predictions_df