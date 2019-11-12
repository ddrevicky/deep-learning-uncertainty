import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from scipy import ndimage
import numpy as np
import PIL
import argparse
import shutil
import random
import io
import sys
from pathlib import Path

from utilities.common_utils import *
from utilities.landmark_utils import *
from utilities.plotting import *
from models import ResUNet, UNet

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser('')
parser.add_argument('--MODE', required=True, type=str, choices=['weight_averaging','mc','ensemble'], help='Evaluation mode.')
parser.add_argument('--MODEL_PATH', required=True, type=str, help='Path to the evaluated model(s).')
parser.add_argument('--DATA_SPLIT', type=str, choices=['train, test1, test2'], default='test1', help='Which data split to evaluate on.')
parser.add_argument('--LOG_PATH', type=str, default='logs', help='Path to model(s).')
parser.add_argument('--SAMPLES', type=int, default=15, help='Number of MC samples to use for prediction.')
parser.add_argument('--IMAGES_PATH', type=str, default='data/images', help='Path to image data.')
parser.add_argument('--ANNOT_PATH', type=str, help='Path to annotation data.')
parser.add_argument('--IMAGE_SIZE', type=int, default=128, help='Size the test images will be rescaled to before being passed to the model.')
parser.add_argument('--GAUSS_SIGMA', type=float, default=5, help='Sigma of the Gaussian kernel used to generate ground truth heatmaps for the landmarks.')
parser.add_argument('--GAUSS_AMPLITUDE', type=float, default=1000.0)
parser.add_argument('--BATCH_SIZE', type=int, default=30)
args = parser.parse_args()


def get_predicted_landmarks(pred_heatmaps, gauss_sigma):
    n_landmarks = pred_heatmaps.shape[0]
    heatmap_y, heatmap_x = pred_heatmaps.shape[1:]
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    for i in range(n_landmarks):
        max_activation, pred_yx = get_max_heatmap_activation(pred_heatmaps[i], gauss_sigma)
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_yx = np.around(pred_yx * rescale)
        pred_landmarks[i] = pred_yx
        max_activations[i] = max_activation
    return pred_landmarks, max_activations


def load_net(path):
    net = torch.load(path)
    net.to(device)
    return net


def enable_test_time_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()


def predict(model_path, test_time_dropout=False):
    # Data frame
    columns = ['file'] + [f'{i}_act' for i in range(N_LANDMARKS)] + \
                         [f'{i}_y' for i in range(N_LANDMARKS)] + \
                         [f'{i}_x' for i in range(N_LANDMARKS)] 
    index = np.arange(n_eval_images)
    df = pd.DataFrame(columns=columns, index=index)
    
    # Model
    net = load_net(model_path)
    
    n_processed = 0
    with torch.no_grad():
        net.eval()
        if test_time_dropout:
            net.apply(enable_test_time_dropout)
        for imgs, true_heatmaps, img_paths in eval_dl:
            imgs = imgs.to(device)
            true_heatmaps = true_heatmaps.to(device)
            pred_heatmaps = net(imgs)
            
            batch_size = pred_heatmaps.shape[0]
            for i in range(batch_size):
                pred_landmarks, max_activations = get_predicted_landmarks(pred_heatmaps[i], args.GAUSS_SIGMA)
                for lm in range(pred_landmarks.shape[0]):
                    row = n_processed + i
                    df.iloc[row]['file'] = Path(img_paths[i]).name
                    df.iloc[row][f'{lm}_act'] = max_activations[lm]
                    df.iloc[row][f'{lm}_y'] = pred_landmarks[lm][0]
                    df.iloc[row][f'{lm}_x'] = pred_landmarks[lm][1]
            n_processed += batch_size
            print(f'Processed {n_processed}/{n_eval_images}')
    return df


log_path = Path(args.LOG_PATH)/f'{args.DATA_SPLIT}'; log_path.mkdir(parents=True, exist_ok=True)
args.MODEL_PATH = Path(args.MODEL_PATH)

# Eval data
data_dir = Path(args.IMAGES_PATH)/f'{args.IMAGE_SIZE}/{args.DATA_SPLIT}'
data_fpaths = list_files(data_dir)    
n_eval_images = len(data_fpaths)
print (f'Generating predictions on data split: {args.DATA_SPLIT}. Number of test images: {n_eval_images}')
eval_ds = LandmarkDataset(data_fpaths, args.ANNOT_PATH, args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE)
eval_dl = DataLoader(eval_ds, args.BATCH_SIZE, shuffle=False, num_workers=1)

model_name = args.MODEL_PATH.stem
if args.MODE == 'weight_averaging':
    log_path = log_path/f'weight_averaging/{model_name}/predictions.csv'; log_path.mkdir(parents=True, exist_ok=True)
    
    print('Weight averaging model prediction.')
    print(f'Log path {log_path}')
    predictions = predict(args.MODEL_PATH, test_time_dropout=False)
    predictions.to_csv(log_path)
elif args.MODE == 'mc':
    log_path = log_path/f'mc/{model_name}/predictions'; log_path.mkdir(parents=True, exist_ok=True)
    
    print(f'MC sampling with {args.SAMPLES} samples.')
    print(f'Log dir:{log_path}')
    for sample in range(args.SAMPLES):
        predictions = predict(args.MODEL_PATH, test_time_dropout=True)
        predictions.to_csv(log_path/f'{sample}.csv')
        print(f'Sample {sample}/{args.SAMPLES} evaluated')
elif args.MODE == 'ensemble':
    model_paths = list_files(args.MODEL_PATH)
    if len(model_paths) > args.SAMPLES:
        model_paths = model_paths[:args.SAMPLES]
    
    log_path = log_path/f'ensemble/{model_name}/predictions'; log_path.mkdir(parents=True, exist_ok=True)
    print('Ensemble prediction.')
    print(f'Log dir:{log_path}')
    for i, model_path in enumerate(model_paths):
        predictions = predict(model_path, test_time_dropout=False)
        model_name = model_path.stem
        predictions.to_csv(log_path/f'{model_name}.csv')
        print(f'Model {i+1}/{len(model_paths)} evaluated')