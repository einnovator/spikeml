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

from spikeml.utils.img_util import show_img, show_imgs

def hist_cov(C, title=None, title_size=6,  ax=None, figsize=(1*3,1*3)):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize) #, dpi=72)
        axs = fig.subplots(1, 1)
        ax = axs
    STYLE_PLOT_TITLE= {'fontsize': title_size }
    if title:
        ax.set_title(title, **STYLE_PLOT_TITLE)
    C_ = C.flatten()
    ax.hist(C_, bins=50, density=True)
    ax.set_xlabel("cov.", fontsize=6)
    ax.set_ylabel("p", fontsize=6)
    ax.set_yscale("log")
    ax.tick_params(axis='both', labelsize=6)
    if fig is not None:
        plt.show()
        
def autocorr2d_fft(img, normalize=True):
    F = np.fft.fft2(img)
    S = np.abs(F)**2
    C = np.fft.ifft2(S).real
    ac2d = np.fft.fftshift(C)  # center at zero displacement
    if normalize:
        ac2d = ac2d/ac2d[img.shape[0]//2, img.shape[1]//2]
    return ac2d

def autocorr_radial_(C):
    ny, nx = C.shape
    y, x = np.indices((ny, nx))
    cy, cx = ny // 2, nx // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    tbin = np.bincount(r.ravel(), C.ravel())
    nr = np.bincount(r.ravel())
    acr = tbin / nr
    return acr

def autocorr_radial(img, normalize=True):
    ac2d = autocorr2d_fft(img, normalize=False)
    acr = autocorr_radial_(ac2d)
    if normalize:
        acr /= acr[0]
    return acr


def autocorr_radial2(f, max_r):
    h, w = f.shape
    C = np.zeros(max_r)
    N = np.zeros(max_r)

    for y in range(h):
        for x in range(w):
            for dy in range(-max_r, max_r+1):
                for dx in range(-max_r, max_r+1):
                    r = int(np.sqrt(dx*dx + dy*dy))
                    if r < max_r:
                        y2, x2 = y+dy, x+dx
                        if 0 <= y2 < h and 0 <= x2 < w:
                            C[r] += f[y, x] * f[y2, x2]
                            N[r] += 1
    return C / N

from mpl_toolkits.axes_grid1 import make_axes_locatable

def show_autocorr2d(ac2d, title='', axs=None, axi=0, figsize=None):
    fig = None
    if axs is None:
        fig = plt.figure(figsize=figsize) #, dpi=72)
        axs = fig.subplots(1, 2)
    ax0 = axs[axi]
    im0 = ax0.imshow(ac2d)
    div0 = make_axes_locatable(ax0)
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    cbar0 = ax0.get_figure().colorbar(im0, cax=cax0, orientation='vertical', label="ac.")
    cbar0.ax.tick_params(labelsize=6)
    ax0.set_title(f"{title}: 2D spatial autocorr.", fontsize=6)
    ax0.tick_params(axis='both', labelsize=6)
    ny, nx = ac2d.shape
    cy, cx = ny // 2, nx // 2
    R = min(ny, nx)//4  # radius in pixels
    ac2d_ = ac2d[cy-R:cy+R+1, cx-R:cx+R+1]
    ax1 = axs[axi+1]
    im1 = ax1.imshow(ac2d_)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    cbar1 = ax1.get_figure().colorbar(im1, cax=cax1, orientation='vertical', label="ac.")
    cbar1.ax.tick_params(labelsize=6)
    ax1.set_title(f"{title}: (zoomed) 2D spatial autocorr.", fontsize=6)
    ax1.tick_params(axis='both', labelsize=6)
    if fig is not None:
        plt.show()
    
def compute_show_autocorr2d(img, normalize=True, title='', axs=None, axi=0, figsize=None):
    ac2d = autocorr2d_fft(img, normalize=normalize)
    show_autocorr2d(ac2d, title=title, axs=axs, axi=axi, figsize=figsize)
    return ac2d

def show_autocorr_radial(acr, title=None, title_size=6, ax=None, figsize=None):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize) #, dpi=72)
        ax = axs = fig.subplots(1, 1)
    #print(acr)
    if title:
        STYLE_PLOT_TITLE= {'fontsize': title_size }
        ax.set_title(title, **STYLE_PLOT_TITLE)
    ax.plot(acr)
    ax.set_xlabel("r", fontsize=6)
    ax.set_ylabel("cor.", fontsize=6)
    ax.tick_params(axis='both', labelsize=6)
    if fig is not None:
        plt.show()
    
def compute_show_autocorr_radial(img, title=None, title_size=6, ax=None, figsize=None):
    acr = autocorr_radial(img)
    #acr2 = autocorr_radial2(img, img.shape[1])
    show_autocorr_radial(acr, title=title, title_size=title_size, ax=ax, figsize=figsize)

def img_cov(a, rescale=False):
    if rescale:
        a = a.copy().astype(float)/255
    a_ = a.flatten()
    A = np.outer(a_, a_)
    return A 

def img_kron(a, rescale=False):
    if rescale:
        a = a.copy().astype(float)/255
    A = np.kron(a, a)
    return A 

def show_img_cov(C, rescale=False, title=None, title_size=6, ax=None, figsize=None):
    if rescale:
        C = (C*255).astype(int)
    show_img(C, title=title, title_size=title_size, ax=ax, figsize=figsize)

def compute_show_img_cov(img, rescale=False, title=None, title_size=6, ax=None, figsize=None):
    C = img_cov(img)
    show_img_cov(C, rescale=rescale, title=title, title_size=title_size, ax=ax, figsize=figsize)
    

def show_img_kron(C, rescale=False, title=None, title_size=6, ax=None, figsize=None):
    if rescale:
        C = (C*255).astype(int)
    show_img(C, title=title, title_size=title_size, ax=ax, figsize=figsize)

def compute_show_img_kron(img, rescale=False, title=None, title_size=6, ax=None, figsize=None):
    C = img_kron(img, rescale=rescale)
    show_img_kron(C, title=title, title_size=title_size, ax=ax, figsize=figsize)


def xshow_img(img, title=f'', cmap=None, bgr=False, title_size=8, axs=None, figsize=None,
    imshow =True, cov_histo=True, ac2d=True, acr=True, cov=True, kron=True):
    fig = None
    ncols = imshow + cov_histo + ac2d*2 + acr + cov + kron
    if axs is None:
        if figsize is None:
            figsize = (ncols*3, 1*3)
        fig = plt.figure(figsize=figsize) #, dpi=72)
        axs = fig.subplots(1, ncols)
    i = 0
    if imshow:
        show_img(img, title=title, cmap=cmap, bgr=bgr, title_size=title_size, ax=axs[i])
        i += 1
    if cov_histo:
        hist_cov(img, title=f'{i}', ax=axs[i])
        i += 1
    if ac2d:
        compute_show_autocorr2d(img, title=f'{i}', axs=axs, axi=i)
        i += 2
    if acr:
        compute_show_autocorr_radial(img, title=f'{i}', title_size=6, ax=axs[i])
        i += 1
    if cov:
        compute_show_img_cov(img, title=f'{i}: cov', ax=axs[i])
        i += 1
    if kron:
        compute_show_img_kron(img, title=f'{i}: kron', ax=axs[i])
        i += 1
    if fig is not None:
        plt.show()
