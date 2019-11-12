import pandas as pd
import numpy as np
import PIL
import argparse
import shutil
import random
import io
import sys
from pathlib import Path
from collections import defaultdict, OrderedDict

from utilities.common_utils import *
from utilities.landmark_utils import *
from utilities.plotting import *
from utilities.eval_utils import *
from models import ResUNet, UNet


parser = argparse.ArgumentParser('')
parser.add_argument('--MODE', type=str, required=True, choices=['mc', 'weight_averaging', 'ensemble'], help='Evaluation mode.')
parser.add_argument('--DATA_SPLIT', type=str, default='test1', choices=['train', 'test1', 'test2'], help='Which data split to evaluate on.')
parser.add_argument('--LOG_PATH', type=str, default='logs', help='Path to model logs.')
parser.add_argument('--SAMPLES', type=int, default=15, help='Number of MC samples to use for prediction.')
parser.add_argument('--MODEL_NAME', type=str, required=True, help='Name of the evaluated model(s).')
parser.add_argument('--ANNOT_PATH', type=str, default='data/AnnotationsByMD/400_senior', help='Path to annotation data.')
parser.add_argument('--IMAGES_PATH', type=str, default='data/images', help='Path to image data.')
parser.add_argument('--IMAGE_SIZE', type=int, default=128, help='Size the test images will be rescaled to before being passed to the model.')
args = parser.parse_args()

args.LOG_PATH = Path(args.LOG_PATH)
args.ANNOT_PATH = Path(args.ANNOT_PATH)

# Get test files
test_dir = Path(args.IMAGES_PATH)/f'{args.IMAGE_SIZE}/{args.DATA_SPLIT}'
test_files = list_files(test_dir)    
n_test = len(test_files)
print(f'Evaluating performance metrics for model {args.MODEL_NAME}')
print(f'Split: {args.DATA_SPLIT}, no. images: {n_test}')

# Compute true landmarks
true_landmarks_dict = OrderedDict()
for i, img_path in enumerate(test_files):
    true_landmarks_dict[img_path.name] = get_true_landmarks(args.ANNOT_PATH, img_path)

# Load pre-generated model predictions from csvs
model_log_dir = args.LOG_PATH/f'{args.DATA_SPLIT}/{args.MODE}/{args.MODEL_NAME}/predictions'
predictions_df = read_prediction_files_as_df(list_files(model_log_dir))

# Compute metrics
radial_errors_all = np.zeros((len(test_files), N_LANDMARKS))
for i, image_file in enumerate(test_files):
    # Compute the statistics across all samples for the image
    landmark_samples, activation_samples = get_predictions_for_image(predictions_df, image_file.name, args.SAMPLES)
    predicted_landmarks_mean, predicted_landmarks_var = get_predicted_landmarks_for_image(landmark_samples)
    activation_mean, activation_var = get_predicted_activations_for_image(activation_samples)

    # Compute radial errors
    true_landmarks = true_landmarks_dict[image_file.name]
    radial_errors_all[i] = get_radial_errors_mm_for_image(true_landmarks, predicted_landmarks_mean)

print(f'Accuracy metrics for model: {args.MODEL_NAME}, test split: {args.DATA_SPLIT}, mode: {args.MODE}, samples: {args.SAMPLES}')
metrics = get_accuracy_metrics(radial_errors_all)
print_accuracy_metrics(metrics)