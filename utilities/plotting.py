import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, FixedLocator, FormatStrFormatter
import PIL
import shutil
from pathlib import Path

from .landmark_utils import *

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_text(ax, pos, label, fontsize, color='#00CC00', outline=2):
    text = ax.text(*pos, label, verticalalignment='top', color=color, fontsize=fontsize)
    draw_outline(text, outline)

def denorm(tensor_im):
    tensor_im = tensor_im / 2.0 + 0.5
    im = tensor_im.numpy()
    im = np.squeeze(im)
    return im

def plot_imgs(imgs, labels=None):
    """ Plots tensor images with annotations.
    """
    # Draw skull with landmarks
    fig, axes = plt.subplots(6, len(imgs) // 6, figsize=(30, 20))
    for i, ax in enumerate(axes.flat):
        if type(imgs[i]) is torch.Tensor:
            imgs[i] = denorm(imgs[i])
        ax.imshow(np.squeeze(imgs[i]), cmap='gray')
        if labels is not None:
            draw_text(ax, (0,0), labels[i], fontsize=17, color='red')
    plt.tight_layout()

def plot_img_with_heatmaps(img, heatmaps, gaussian_sigma):
    """ Plots tensor images with annotations.
    """
    # Draw skull with landmarks
    skull_fig, ax = plt.subplots(1, 1, figsize=(8, 8), frameon=False)
    ax.set_axis_off()
    if type(img) is torch.Tensor:
        img = denorm(img)
    ax.imshow(img, cmap='gray')
    for i, heatmap in enumerate(heatmaps):
        if i < N_LANDMARKS:
            _, annot_yx = get_max_heatmap_activation(heatmap, gaussian_sigma)
            annot_xy = annot_yx[1], annot_yx[0]
            draw_annot(ax, annot_xy, label=str(i+1), radius=0.5, fontsize=12)
    
    # Draw heatmaps
    fig, axes = plt.subplots(4, 5, figsize=(12, 8))    
    axes[-1,-1].imshow(np.zeros_like(heatmaps[0]))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < N_LANDMARKS:
            ax.imshow(heatmaps[i])
            _, annot_yx = get_max_heatmap_activation(heatmaps[i], gaussian_sigma)
            annot_xy = annot_yx[1], annot_yx[0]
    plt.tight_layout()
    return skull_fig

def draw_annot(ax, pos, label, radius=40, fontsize=18, 
               color='red', marker='x', alpha=1.0):
    if marker is not None:
        ax.scatter(pos[0], pos[1], s=radius, c=color, marker=marker, alpha=alpha)
    if label is not None:
        draw_text(ax, pos, label, fontsize, color=color)