import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import PIL
import shutil
from pathlib import Path
import time
import random

from .common_utils import *

# Parameter for gaussian filter
GAUSSIAN_TRUNCATE = 1.0

N_LANDMARKS = 19

def get_annots_for_image(annotations_path, image_path, rescaled_image_size=None, orig_image_size=np.array([ORIG_IMAGE_X, ORIG_IMAGE_Y])):
    image_id = image_path.stem
    annots = (annotations_path/f'{image_id}.txt').read_text()
    annots = annots.split('\n')[:N_LANDMARKS]
    annots = [l.split(',') for l in annots]
    annots = [(float(l[0]), float(l[1])) for l in annots];
    annots = np.array(annots)
    if rescaled_image_size is not None:
        scale = np.array([rescaled_image_size, rescaled_image_size], dtype=float) / orig_image_size # WxH
        annots = np.around(annots * scale).astype('int32')
    return annots # [[x1,y1], [x2,y2], ...]

def create_true_heatmaps(annots, image_size, amplitude):
    heatmaps = np.zeros((annots.shape[0], image_size, image_size))
    for i, landmark_pos in enumerate(annots):
        x, y = landmark_pos
        heatmaps[i, y, x] = amplitude # Swap WxH to HxW
    return heatmaps

def reset_heatmap_maximum(heatmap, amplitude):
    '''Heatmap maximum value is not equal to the amplitude after the transformation.
       We zero the heatmap and set it to the amplitude at the new maximum position.
    '''
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    heatmap[:] = 0
    heatmap[ind] = amplitude
    return heatmap

class ArrayToTensor(object):
    def __call__(self, np_array):
        return torch.from_numpy(np_array).float()
    
class LandmarkDataset(Dataset):
    def __init__(self, image_fnames, annotations_path, gauss_sigma, gauss_amplitude, 
                 elastic_trans=None, affine_trans=None, horizontal_flip=False):
        self.image_fnames = image_fnames
        if annotations_path == '': annotations_path = None
        self.annotations_path = annotations_path
        self.gauss_sigma = gauss_sigma
        self.gauss_amplitude = gauss_amplitude
        self.elastic_trans = elastic_trans
        self.affine_trans = affine_trans
        self.horizontal_flip = horizontal_flip
        
    def __len__(self):
        return len(self.image_fnames)
        
    def __getitem__(self, idx):
        # So that each thread has a different transform when multiprocessing
        seed = int(random.random() * 10000000)
        np.random.seed(seed)
        
        # Image
        x = PIL.Image.open(self.image_fnames[idx]).convert('L')
        image_size = x.size[0]
        x = np.array(x)
        
        if self.affine_trans is not None:
            do_affine = np.random.uniform() > 0.02
            if do_affine:
                affine_matrix = self.affine_trans.get_matrix(x)
                affine_matrix = np.linalg.inv(affine_matrix)
        
        # Elastic transform
        if self.elastic_trans is not None:
            #do_elastic = np.random.uniform() > 0.1
            do_elastic = True
            if do_elastic:
                x_coords, y_coords, _, _ = self.elastic_trans.get_coordinates(x)
                elastic_trans_coordinates = (y_coords, x_coords)
        
        if self.horizontal_flip:
            do_flip = np.random.uniform() > 0.5
        
        # x transforms
        if self.elastic_trans is not None and do_elastic:
            x = ndimage.interpolation.map_coordinates(x, elastic_trans_coordinates, order=1).reshape(x.shape)
        if self.affine_trans is not None and do_affine:
            x = ndimage.affine_transform(x, affine_matrix, offset=0, order=1)
        if self.horizontal_flip and do_flip:
            x = np.ascontiguousarray(np.flip(x, axis=1))
        x = np.expand_dims(x, 2)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
        
        # Labels
        # Annotations
        if self.annotations_path is not None:
            annots = get_annots_for_image(self.annotations_path, self.image_fnames[idx], rescaled_image_size=image_size)
            if self.horizontal_flip and do_flip:
                for i in range(annots.shape[0]):
                    annots[i][0] += 2*(image_size/2.0 - annots[i][0])

            # Create unfiltered heatmaps
            y = create_true_heatmaps(annots, image_size, amplitude=self.gauss_amplitude)

            if self.elastic_trans is not None and do_elastic:
                for i in range(y.shape[0]):
                    y[i] = ndimage.interpolation.map_coordinates(y[i], elastic_trans_coordinates, order=1).reshape(y[i].shape)
                    y[i] = reset_heatmap_maximum(y[i], self.gauss_amplitude)

            if self.affine_trans is not None and do_affine:
                for i in range(y.shape[0]):
                    y[i] = ndimage.affine_transform(y[i], affine_matrix, offset=0, order=1)
                    y[i] = reset_heatmap_maximum(y[i], self.gauss_amplitude)

            # Apply gaussian filter
            for i in range(y.shape[0]):
                y[i] = ndimage.gaussian_filter(y[i], sigma=self.gauss_sigma, truncate=GAUSSIAN_TRUNCATE)
        else:
            y = np.array([0])
        
        y = torch.from_numpy(y).float()
        return x, y, str(self.image_fnames[idx])

'''
def get_max_yx(tensor):
    max_y, argmax_y = tensor.max(dim=0)
    _, argmax_x = max_y.max(dim=0)
    max_yx = (argmax_y[argmax_x.item()].item(), argmax_x.item())
    return np.array(max_yx)
'''

def np_max_yx(arr):
    argmax_0 = np.argmax(arr, axis=0)
    max_0 = arr[argmax_0, np.arange(arr.shape[1])]
    argmax_1 = np.argmax(max_0)
    max_yx_pos = np.array([argmax_0[argmax_1], argmax_1])
    max_val = arr[max_yx_pos[0], max_yx_pos[1]]
    return max_val, max_yx_pos

def get_max_heatmap_activation(tensor, gauss_sigma):
    array = tensor.cpu().detach().numpy()
    activations = ndimage.gaussian_filter(array, sigma=gauss_sigma, truncate=GAUSSIAN_TRUNCATE)
    max_val, max_pos = np_max_yx(activations)
    return max_val, max_pos

def radial_errors_example(pred, targ, gauss_sigma, orig_image_x=ORIG_IMAGE_X, orig_image_y=ORIG_IMAGE_Y):
    example_radial_errors = np.zeros(N_LANDMARKS)
    heatmap_y, heatmap_x = pred.shape[1:]
    for i in range(N_LANDMARKS):
        max_pred_act, pred_yx = get_max_heatmap_activation(pred[i], gauss_sigma)
        _, true_yx = get_max_heatmap_activation(targ[i], gauss_sigma)
        
        # Rescale to original resolution
        rescale = np.array([ORIG_IMAGE_Y, ORIG_IMAGE_X]) / np.array([heatmap_y, heatmap_x])
        pred_yx = np.around(pred_yx * rescale) / PIXELS_PER_MM
        true_yx = np.around(true_yx * rescale) / PIXELS_PER_MM
        example_radial_errors[i] = np.linalg.norm(pred_yx - true_yx)
    return example_radial_errors

def radial_errors_batch(preds, targs, gauss_sigma):
    assert(preds.shape[0] == targs.shape[0])
    batch_size = preds.shape[0]
    batch_radial_errors = np.zeros((batch_size, N_LANDMARKS))
    for i in range(batch_size):
        batch_radial_errors[i] = radial_errors_example(preds[i], targs[i], gauss_sigma)
    return batch_radial_errors