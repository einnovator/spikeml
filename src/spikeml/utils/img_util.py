import sys
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import random
import math
import copy
from typing import Annotated, Any, Callable

import cv2
from PIL import Image

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

def show_img(img, title=None, cmap=None, bgr=False, title_size=8, ax=None, figsize=None, aspect='auto', callback=None):
    if 'torch' in sys.modules and torch.is_tensor(img):
        if len(img.shape)==2:
            img= img.unsqueeze(dim=0)
        img = img.permute(1, 2, 0)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)#, dpi=72)
        ax = fig.subplots(1, 1)
    ax.axis('off')
    if title:
        STYLE_PLOT_TITLE= {'fontsize': title_size }
        ax.set_title(title, **STYLE_PLOT_TITLE)
    if cmap is None:
        cmap = 'gray' if len(img.shape)==2 else None
    if bgr and len(img.shape)==3 and img.shape[2]==3:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = ax.imshow(img, cmap=cmap, aspect=aspect)
    if callback is not None:
        callback(ax, im)
    if fig is not None:
        plt.show()
    return ax

DEFAULT_CMAP='viridis'

def mshow(img, title=None, cmap=DEFAULT_CMAP, title_size=8, labels_size=6, ax=None, figsize=None,
          colorbar=False, colorbar_label=None, colorbar_labelsize=6, colorbar_fontsize=6, aspect='auto',
          row_names=None, col_names=None, rotation=45, callback=None):
    
    def _decor(ax, im):
        ax.axis('on')
        if row_names is not None and len(row_names)>0:
            ax.set_yticks(np.arange(len(row_names)))
            ax.set_yticklabels(row_names)
            ax.set_yticklabels(row_names, fontsize=labels_size)
            print(row_names)
        if col_names is not None and len(col_names)>0:
            ax.set_xticks(np.arange(len(col_names)))
            ax.set_xticklabels(col_names)
            ax.set_xticklabels(col_names, fontsize=labels_size)
            plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
            print(col_names)
        if colorbar:
            cbar = ax.figure.colorbar(im, ax=ax)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label, fontsize=colorbar_labelsize)
            cbar.ax.tick_params(labelsize=colorbar_fontsize)
            
        if callback is not None:
            callback(ax, im)


    show_img(img, title=title, cmap=cmap, title_size=title_size, ax=ax, figsize=figsize, aspect=aspect,
             callback=_decor)
    


#imgs: [array(H,W)[:int32]]
def show_imgs(imgs, titles=None, suptitle=None, cmap=None, ncols=None, nrows=None, bgr=False,
              suptitle_size=6, title_size=6, figsize=None, pad=0, normalize=True):
    if isinstance(imgs, list):
        imgs = np.array(imgs)
    n = imgs.shape[0]
    if ncols==None:
        if nrows==None:
            ncols=1
        else:
            ncols=int(n/nrows) + (n % nrows > 0)
    if nrows==None:
        nrows=n//ncols + int(n% ncols!=0)
    fig = plt.figure(figsize=figsize) #, dpi=72)
    axs = fig.subplots(nrows, ncols)
    if nrows>1 or ncols>1:
        axs = axs.flatten()
    else:
        axs = [axs]
    plt.subplots_adjust(top=1-pad, bottom=0+pad, left=0+pad, right=1-pad, wspace=pad, hspace=pad) 
    #print('!!',nrows, ncols, axs)
    for ax in axs:
        ax.axis('off')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(left=False)
    for i in range(imgs.shape[0]):
        img = imgs[i] 
        if cmap is None:
            cmap_ = 'gray' if len(img.shape)==2 else None
        else:
            cmap_ = cmap[i] if isinstance(cmap, list) else cmap
        if bgr and len(img.shape)==3 and img.shape[2]==3:
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not normalize:
            axs[i].imshow(img, cmap=cmap_, vmin=0, vmax=img.max())
        else:
            axs[i].imshow(img, cmap=cmap_)
        STYLE_PLOT_TITLE= {'fontsize': title_size }
        if titles and i<len(titles):
            axs[i].set_title(str(titles[i]), **STYLE_PLOT_TITLE)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=suptitle_size, y=1.1)
    plt.show()
    return axs

    
def cv_bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_kernels(M, shape=None, figsize=None, pad=.1,  titles=None, suptitle=None, suptitle_size=6, title_size=6,
                 normalize=True):
    if shape is not None:
        M = M.reshape(M.shape[0], shape[0], shape[1])
    if titles is None:
        titles = [f'{i}' for i in range(0, M.shape[0])]        
    show_imgs(M, ncols=min(M.shape[0],20), figsize=figsize, pad=pad,
              titles=titles, suptitle=suptitle, suptitle_size=suptitle_size, title_size=title_size, normalize=normalize)


def show_kernels_(M__, step=10, figsize=None):
    for t in range(0, len(M__)):
        if t % step != 0:
            continue
        show_imgs(M__[t], ncols=min(M.shape[0],20), title_size=6, figsize=figsize, titles=[f'{t}:{i}' for i in range(0, M__[0].shape[0])])
        #print(M__[t][0])