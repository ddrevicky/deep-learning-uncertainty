import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

ORIG_IMAGE_SIZE = np.array([ORIG_IMAGE_X, ORIG_IMAGE_Y]) # WxH

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_PATH', type=str, default='./data', help='Root path to the data.')
parser.add_argument('--MODEL_PATH', type=str, default='./trained', help='Path where the model checkpoints will be saved.')
random_id = int(random.uniform(0, 99999999))
parser.add_argument('--MODEL_NAME', type=str, default=f'model_{random_id}')
parser.add_argument('--EXPERIMENT_NAME', type=str, default=f'exp_{random_id}')
parser.add_argument('--MODEL', type=str, default='unet')
parser.add_argument('--FILTERS', type=lambda layers: [int(layer) for layer in layers.split(',')], default='64,128,256,512,1024')
parser.add_argument('--DOWN_DROP', type=lambda layers: [float(layer) for layer in layers.split(',')], default='0.4,0.4,0.4,0.4')
parser.add_argument('--UP_DROP', type=lambda layers: [float(layer) for layer in layers.split(',')], default='0.4,0.4,0.4,0.4')
parser.add_argument('--BATCH_SIZE', type=int, default=8)
parser.add_argument('--IMAGE_SIZE', type=int, default=128)
parser.add_argument('--GAUSS_SIGMA', type=float, default=5.0)
parser.add_argument('--GAUSS_AMPLITUDE', type=float, default=1000.0)
parser.add_argument('--USE_ELASTIC_TRANS', type=bool, default=False)
parser.add_argument('--USE_AFFINE_TRANS', type=bool, default=False)
parser.add_argument('--USE_HORIZONTAL_FLIP', type=bool, default=False)
parser.add_argument('--ELASTIC_SIGMA', type=float, default=10.0)
parser.add_argument('--ELASTIC_ALPHA', type=float, default=15.0)
parser.add_argument('--LEARN_RATE', type=float, default=1e-3)
parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0)
parser.add_argument('--OPTIM_PATIENCE', type=float, default=15)
parser.add_argument('--EPOCHS', type=int, default=400)
parser.add_argument('--VALID_RATIO', type=float, default=0.15)
parser.add_argument('--SAVE_EPOCHS', type=lambda epochs: [float(epoch) for epoch in epochs.split(',')], default=None)
parser.add_argument('--VAL_MRE_STOP', type=float, default=None,
                   help='Stops training if validation MRE drops below the specified value.')
args = parser.parse_args()

print(f'Training model {args.MODEL_NAME}')

# Data paths
path = Path(args.DATA_PATH)
annotations_path = path/'AnnotationsByMD'/'400_senior'
model_path = Path(args.MODEL_PATH) if args.MODEL_PATH is not None else path/'models'
model_path.mkdir(parents=True, exist_ok=True)
train_path = path/f'images/{args.IMAGE_SIZE}/train'

# Datasets, DataLoaders
fnames = list_files(train_path)
n_valid = int(args.VALID_RATIO * len(fnames))
train_fnames = fnames[:-n_valid]
valid_fnames = fnames[-n_valid:]
print (f'Number of train images: {len(train_fnames)}, Number of validation images: {len(valid_fnames)}')

num_workers = 8
elastic_trans = None
affine_trans = None
if args.USE_ELASTIC_TRANS:
    elastic_trans = ElasticTransform(sigma=args.ELASTIC_SIGMA, alpha=args.ELASTIC_ALPHA)
if args.USE_AFFINE_TRANS:
    angle = 5
    scales = [0.95, 1.05]
    tx, ty = 0.03, 0.03
    affine_trans = AffineTransform(angle, scales, tx, ty)
train_ds = LandmarkDataset(train_fnames, annotations_path, args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE, elastic_trans=elastic_trans,
                          affine_trans=affine_trans, horizontal_flip=args.USE_HORIZONTAL_FLIP)
train_dl = DataLoader(train_ds, args.BATCH_SIZE, shuffle=True, num_workers=num_workers)
valid_ds = LandmarkDataset(valid_fnames, annotations_path, args.GAUSS_SIGMA, args.GAUSS_AMPLITUDE)
valid_dl = DataLoader(valid_ds, args.BATCH_SIZE, shuffle=False, num_workers=num_workers)

# Net model
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if args.MODEL == 'unet':
    net = UNet(in_ch=1, out_ch=N_LANDMARKS, down_drop=args.DOWN_DROP, up_drop=args.UP_DROP)
net.to(device);

# Optimizer + loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.LEARN_RATE, weight_decay=args.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.OPTIM_PATIENCE, verbose=True)


def train():    
    train_loss, train_mre, train_sdr_4mm = 0, 0, 0
    train_examples = 0
    net.train()
    for imgs, true_heatmaps, _ in train_dl:
        imgs = imgs.to(device)
        true_heatmaps = true_heatmaps.to(device)
        
        optimizer.zero_grad()
        pred_heatmaps = net(imgs)
        loss = criterion(pred_heatmaps, true_heatmaps)
        loss.backward()
        optimizer.step()
        
        # Metrics
        actual_bs = imgs.shape[0]
        train_loss += loss * actual_bs  # Weighted by batch size
        train_examples += actual_bs
        
        radial_errors = radial_errors_batch(pred_heatmaps, true_heatmaps, args.GAUSS_SIGMA)
        mre = np.mean(radial_errors)
        train_mre += mre * actual_bs
        train_sdr_4mm += np.sum(radial_errors < 4)
    
    print(f'Epoch: {e}, train_loss: {train_loss / train_examples:{4}.{4}}, '
          f'train_MRE: {train_mre / train_examples:{4}.{4}}, '
          f'train_SDR_4mm: {train_sdr_4mm/(train_examples * N_LANDMARKS):{4}.{4}}, ', end='')
    return train_loss / train_examples
    
    
def validate():
    val_loss, val_mre, val_sdr_4mm = 0, 0, 0
    val_examples = 0
    net.eval()
    with torch.no_grad():
        for imgs, true_heatmaps, _ in valid_dl:
            imgs = imgs.to(device)
            true_heatmaps = true_heatmaps.to(device)

            pred_heatmaps = net(imgs)
            loss = criterion(pred_heatmaps, true_heatmaps)

            actual_bs = imgs.shape[0]
            val_loss += loss * actual_bs
            val_examples += actual_bs
            
            radial_errors = radial_errors_batch(pred_heatmaps, true_heatmaps, args.GAUSS_SIGMA)
            mre = np.mean(radial_errors)
            val_mre += mre * actual_bs
            val_sdr_4mm += np.sum(radial_errors < 4)
        
    print(f'val_loss: {val_loss / val_examples:{4}.{4}}, '
          f'val_MRE: {val_mre / val_examples:{4}.{4}}, '
          f'val_SDR_4mm: {val_sdr_4mm/(val_examples * N_LANDMARKS):{4}.{4}}')
    return val_loss / val_examples, val_sdr_4mm / (val_examples * N_LANDMARKS), val_mre / val_examples


# Loop over epochs
best_val_loss = None
num_bad_epochs = 0
try:
    for e in range(1, args.EPOCHS+1):
        train_loss = train()
        val_loss, val_sdr_4mm, val_mre = validate()
        scheduler.step(val_loss)
        
        if args.SAVE_EPOCHS is not None and e in args.SAVE_EPOCHS:
            print(f'Saving model checkpoint (one of {args.SAVE_EPOCHS} requested).')
            with open(model_path/f'{args.MODEL_NAME}_e_{e}.pth', 'wb') as f:
                torch.save(net, f)
        
        if not best_val_loss or val_loss < best_val_loss * 0.9999 :
            num_bad_epochs = 0
            best_val_loss = val_loss
            checkpoint_path = model_path/f'{args.MODEL_NAME}.pth'
            print(f'Saving model checkpoint to {checkpoint_path}.')
            with open(checkpoint_path, 'wb') as f:
                torch.save(net, f)
                
        if args.VAL_MRE_STOP is not None and val_mre < args.VAL_MRE_STOP:
            print(f'Stopping experiment since val_mre is {val_mre} which is below {args.VAL_MRE_STOP} as specified by program arguments.')
            break
                
        if val_loss > best_val_loss:
            num_bad_epochs += 1
        
        if num_bad_epochs > 30:
            print('Stopping experiment due to plateauing.')
            break
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')