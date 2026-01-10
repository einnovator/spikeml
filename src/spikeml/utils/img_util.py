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

def show_img(img, title=None, cmap=None, bgr=False, title_size=8, ax=None, figsize=None):
    if 'torch' in sys.modules and torch.is_tensor(img):
        if len(img.shape)==2: img= img.unsqueeze(dim=0)
        img = img.permute(1, 2, 0)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)#, dpi=72)
        ax = axs = fig.subplots(1, 1)
    ax.axis('off')
    STYLE_PLOT_TITLE= {'fontsize': title_size }
    if title: ax.set_title(title, **STYLE_PLOT_TITLE)
    if cmap is None:
        cmap = 'gray' if len(img.shape)==2 else None
    if bgr and len(img.shape)==3 and img.shape[2]==3:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap=cmap)
    if fig is not None:
        plt.show()
    return ax

#imgs: [array(H,W)[:int32]]
def show_imgs(imgs, titles=None, cmap=None, ncols=None, nrows=None, bgr=False, title_size=4, figsize=None, pad=0):
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
        axs[i].imshow(img, cmap=cmap_)
        STYLE_PLOT_TITLE= {'fontsize': title_size }
        if titles and i<len(titles):
            axs[i].set_title(str(titles[i]), **STYLE_PLOT_TITLE)
    plt.show()
    return axs

    
def cv_bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)