import numpy as np
from enum import Enum, auto
import cv2
import random

from spikeml.utils.img_util import show_img, show_imgs, cv_bgr2rgb

class Glyph(Enum):
    FILL = auto()
    VSLINE = auto()
    VLINE = auto()
    HSLINE = auto()
    HLINE = auto()
    PLUS = auto()
    CROSS = auto()
    SLASH = auto()
    BACKSLASH = auto()
    CORNER_TL = auto()
    CORNER_TR = auto()
    CORNER_BL = auto()
    CORNER_BR = auto()
    SQUARE = auto()
    BLOCK = auto()
    TRIANGLE = auto()
    INV_TRIANGLE = auto()
    CIRC = auto()
    LEFT = auto()
    RIGHT = auto() 
    TOP = auto()
    BOTTOM = auto()   
    LEFT3 = auto()
    RIGHT3 = auto() 
    TOP3 = auto()
    BOTTOM3 = auto()   

    def __str__(self):
        return self.name.lower().capitalize()

def glyph(n=0, shape=(28,28), color=(255,255,255), lw=1, dx=0, dy=0, sx=1, sy=1, rot=0, rescale=True):
    return make_glyph(n, shape=shape, color=color, lw=lw, dx=dx, dy=dy, sx=sx, sy=sy, rot=rot, rescale=rescale)
    
def make_glyph(n=0, shape=(28,28), color=(255,255,255), lw=1, dx=0, dy=0, sx=1, sy=1, rot=0, rescale=True):
    if not isinstance(shape, tuple):
        shape = (shape, shape)        
    a = np.zeros(shape, dtype=np.uint8)
    w,h = a.shape[1],a.shape[0]
    if isinstance(n, int):
        n = list(Glyph)[n]
    if n==Glyph.FILL: 
        cv2.rectangle(a, (0,0), (w,h), color, -1)
    elif n==Glyph.VSLINE:
        cv2.line(a,(w//2,h//4),(w//2,h//4*3),color,lw)
    elif n==Glyph.VLINE:
        cv2.line(a,(w//2,0),(w//2,h),color,lw)
    elif n==Glyph.HSLINE:
        cv2.line(a,(w//4,h//2),(w//4*3,h//2),color,lw)
    elif n==Glyph.HLINE:
        cv2.line(a,(0,h//2),(w,h//2),color,lw)
    elif n==Glyph.PLUS:
        cv2.line(a,(w//2,h//4),(w//2,h//4*3),color,lw)
        cv2.line(a,(w//4,h//2),(w//4*3,h//2),color,lw)
    elif n==Glyph.CROSS:
        cv2.line(a,(w//2,0),(w//2,h),color,lw)
        cv2.line(a,(0,h//2),(w,h//2),color,lw)
    elif n==Glyph.SLASH:
        cv2.line(a,(3*w//4,h//4),(w//4,3*h//4),color,lw)
    elif n==Glyph.BACKSLASH: #GLYPH_
        cv2.line(a,(w//4,h//4),(3*w//4,3*h//4),color,lw)
    elif n==Glyph.CORNER_TL: #GLYPH_
        cv2.line(a,(w//4,h//4),(3*w//4,h//4),color,lw)
        cv2.line(a,(w//4,h//4),(w//4,3*h//4),color,lw)
    elif n==Glyph.CORNER_TR: 
        cv2.line(a,(w//4,h//4),(3*w//4,h//4),color,lw)
        cv2.line(a,(3*w//4,h//4),(3*w//4,3*h//4),color,lw)
    elif n==Glyph.CORNER_BL:
        cv2.line(a,(w//4,3*h//4),(3*w//4,3*h//4),color,lw)
        cv2.line(a,(w//4,h//4),(w//4,3*h//4),color,lw)
    elif n==Glyph.CORNER_BR:
        cv2.line(a,(w//4,3*h//4),(3*w//4,3*h//4),color,lw)
        cv2.line(a,(3*w//4,h//4),(3*w//4,3*h//4),color,lw)
    elif n==Glyph.SQUARE: 
        cv2.line(a,(w//4,h//4),(3*w//4,h//4),color,lw)
        cv2.line(a,(w//4,3*h//4),(3*w//4,3*h//4),color,lw)
        cv2.line(a,(w//4,h//4),(w//4,3*h//4),color,lw)
        cv2.line(a,(3*w//4,h//4),(3*w//4,3*h//4),color,lw)
    elif n==Glyph.BLOCK:
        cv2.rectangle(a, (w//4,h//4), (3*w//4,3*h//4), color=color, thickness=-1)
    elif n==Glyph.TRIANGLE:
        cv2.line(a,(w//4,3*h//4),(3*w//4,3*h//4),color,lw)
        cv2.line(a,(w//4,3*h//4),(w//2,h//4),color,lw)
        cv2.line(a,(3*w//4,3*h//4),(w//2,h//4),color,lw)
    elif n==Glyph.INV_TRIANGLE:
        cv2.line(a,(w//4,h//4),(3*w//4,h//4),color,lw)
        cv2.line(a,(w//4,h//4),(w//2,3*h//4),color,lw)
        cv2.line(a,(3*w//4,h//4),(w//2,3*h//4),color,lw)
    elif n==Glyph.CIRC:
        cv2.circle(a,(w//2,h//2), 2*min(w, h)//6, color, lw)
    elif n==Glyph.LEFT:        
        cv2.rectangle(a, (0, 0), (w//2,h), color, -1)
    elif n==Glyph.RIGHT:        
        cv2.rectangle(a, (w//2,0), (w,h), color, -1)
    elif n==Glyph.TOP:        
        cv2.rectangle(a, (0, 0), (w,h//2), color, -1)
    elif n==Glyph.BOTTOM:        
        cv2.rectangle(a, (w//2, h//2), (w,h), color, -1)
    elif n==Glyph.LEFT3:        
        cv2.rectangle(a, (0, 0), (w//3,h), color, -1)
    elif n==Glyph.RIGHT3:        
        cv2.rectangle(a, (w//3,0), (w,h), color, -1)
    elif n==Glyph.TOP3:        
        cv2.rectangle(a, (0, 0), (w,h//3), color, -1)
    elif n==Glyph.BOTTOM3:        
        cv2.rectangle(a, (w//3, h//3), (w,h), color, -1)
    else:
        print(f'WARN: unknown shape: {n}')
        return a
        
    if dx!=0 or dy!=0:
        dxy = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(a, dxy, (w, h))
    if sx!=1 or sy!=1:
        a = cv2.resize(a, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA) #| cv2.INTER_CUBIC | INTER_LINEAR
    if rot!=0:
        rm = cv2.getRotationMatrix2D((w//2,h//2), rot, 1.0)
        a = cv2.warpAffine(a, rm, (w, h))
    if rescale:
       a = a.astype(float)/255
    return a

def make_glyphs(nn=None, shape=(28,28), color=(255,255,255), lw=1, dx=0, dy=0, sx=1, sy=1, rot=0, rescale=True):
    if nn is None:
        nn = list(Glyph)
    return [ make_glyph(n, shape=shape, color=color, lw=lw, dx=dx, dy=dy, sx=sx, sy=sy, rot=rot, rescale=rescale) for n in nn ] 


def show_glyphs(nn=None, shape=(28,28), color=(255,255,255), lw=1, dx=0, dy=0, sx=1, sy=1, rot=0, rescale=True, ncols=10, pad=1):
    if nn is None:
        nn = list(Glyph)
    ncols = min(ncols, len(nn))
    nrows = len(nn) // ncols + int(len(nn)%ncols!=0)
    if len(nn)>0 and not isinstance(nn[0], np.ndarray):
        aa = make_glyphs(nn, shape=shape, color=color, lw=lw, dx=dx, dy=dy, sx=sx, sy=sy, rot=rot, rescale=rescale)
        titles = [ str(g) for g in nn ]
    else:
        aa = nn
        titles = None
    show_imgs(aa, ncols=10, titles=titles, title_size=6, figsize=(ncols*1, nrows*1))


def glyph_dataset(n, gg=None, shape=(28,28), color=(255,255,255), lw_min=1, lw_max=3, 
            txs=True, unique=False, shuffle=True,
            dx_min=-.2, dx_max=.2, dy_min=-.2, dy_max=.2, sx_min=.8, sx_max=1.2, sy_min=.8, sy_max=1.2, 
            rot_min=-10, rot_max=10, rescale=True):
    if gg is None:
        gg = list(Glyph)
    xx = []
    yy = []
    dd = []
    gg_ = set()
    if not txs:
        n = min(n,len(gg))
    y = 0
    while len(xx)<n:
        if shuffle:
            y = random.randrange(0, len(gg))
            if unique:
                if y in gg_:
                    if len(gg_)==len(gg):
                        break
                    continue
                gg_.add(y)
            g = gg[y]
        else:
            g = gg[y]
        if txs:
            lw_min = max(min(1, int(min(shape[0],shape[1])*lw_min)), a.shape[0]//4) if isinstance(lw_min, float) else lw_min
            lw_max = max(min(1, int(min(shape[0],shape[1])*lw_max)), a.shape[0]//4) if isinstance(lw_max, float) else lw_max
            lw = random.randrange(lw_min, lw_max)
            dx_min = int(shape[1]*dx_min) if isinstance(dx_min, float) else dx_min
            dx_max = int(shape[1]*dx_max) if isinstance(dx_max, float) else dx_max
            dy_min = int(shape[1]*dy_min) if isinstance(dy_min, float) else dy_min
            dy_max = int(shape[1]*dy_max) if isinstance(dy_max, float) else dy_max
            dx = random.randrange(dx_min, dx_max)
            dy = random.randrange(dy_min, dy_max)
            sx = sx_min + random.random()* (sx_max-sx_min)
            sy = sx_min + random.random()* (sy_max-sy_min)
            rot = rot_min + random.random() * (rot_max-rot_min)        
        else:
            lw = 1
            dx = dy = 0
            sx = sy = 1
            rot = 0
        a = make_glyph(g, shape=shape, color=color, lw=lw, dx=dx, dy=dy, sx=sx, sy=sy, rot=rot, rescale=rescale)
        xx.append(a)
        yy.append(y)
        dd.append({'label': y, 'id': g, 'color': color, 'lw': lw, 'dx':dx, 'dy':dy, 'sx':sx, 'sy': sy, 'rot': rot})
        if not shuffle:
            y += 1
            y %= len(gg)

    xx = np.array(xx)
    yy = np.array(yy)
    return xx, yy, gg, dd


def show_glyph_txs(i, rot=True, sxy=True, dx=True, dy=True,
                   imshow =True, cov_histo=True, ac2d=True, acr=True, cov=True, kron=True):
    def _values(x, value0, value1):
        if isinstance(x, list):
            return x
        if isinstance(x, bool):
            return value1 if x else value0
        return [x]
    rot = _values(rot, [0], [0, -20, +20])
    sxy = _values(sxy, [1], [1, 1.5])
    dx = _values(dx, [0], [0, 10])
    dy = _values(dy, [0], [0, 10])
    for rot_ in rot:
        for sxy_ in sxy:
            for dx_ in dx:
                for dy_ in dy:
                    a = make_glyph(i, dx=dx_, dy=dy_, sx= sxy_, sy=sxy_, rot=rot_)
                    xshow_img(a, title=f'{i} rot({rot_}) s({sxy_}) d({dx_},{dy_}) ', title_size=8,
                    imshow=imshow, cov_histo=cov_histo, ac2d=ac2d, acr=acr, cov=cov, kron=kron)

def show_all_glyph_txs(rot=False, sxy=False, dx=False, dy=False):
    for g in Glyph:
        show_glyph_txs(g, rot=rot, sxy=sxy, dx=dx, dy=dy)
    
