import sys
if __name__ == '__main__':  
    sys.path.insert(0, '.')
    sys.path.insert(0, '..')

import io
import cv2

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import util.geom_util

WHITE=(255,255,255)
WHITE0=(245,117,16)
BLUE=(255,0,0)
BLACK=(0,0,0)
YELLOW=(0,255,255)
GREEN=(0,255,0)
CYAN=(255,255,0)
ORANGE=(0,215,255)
GREY5=(8,8,8)
GREY4=(16,16,16)
GREY3=(32,32,32)
GREY2=(64,64,64)
GREY1=(128,128,128)

FONT_SIZE_NORMAL = 1
FONT_SIZE_SMALLER = 0.6
FONT_SIZE_SMALL = 0.5
FONT_SIZE_XSMALL = 0.4
FONT_SIZE_XXSMALL = 0.3


FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
FONT_HERSHEY_PLAIN = cv2.FONT_HERSHEY_PLAIN
FONT_HERSHEY_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
FONT_HERSHEY_COMPLEX = cv2.FONT_HERSHEY_COMPLEX
FONT_HERSHEY_TRIPLEX = cv2.FONT_HERSHEY_TRIPLEX
FONT_HERSHEY_COMPLEX_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_HERSHEY_SCRIPT_SIMPLEX = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_HERSHEY_SCRIPT_COMPLEX = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

def draw_rect(img, p0, p1=None, size=None, color=(255,255,555), alpha=0.5):
    (x,y) = p0
    if size==None:
        (x1,y0) = p1
        (w,h) = (x1-x, y0-y)
    else:
        (w,h) = size
    img_ = img[y:y+h, x:x+w]
    bgbox = np.ones(img_.shape, dtype=np.uint8) * 255
    alpha = max(0, min(1, alpha))
    out = cv2.addWeighted(img_, 1-alpha, bgbox, alpha, 1.0)
    img[y:y+h, x:x+w] = out
    return img

def draw_text(image, s, xy, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=FONT_SIZE_NORMAL, thickness=1, color=BLACK):
    cv2.putText(image, str(s), xy, font, font_size, color, thickness, cv2.LINE_AA)

def draw_text_boxed(image, s, xy, min_width=None, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=FONT_SIZE_NORMAL, thickness=1, color=BLACK, bgcolor=WHITE):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = FONT_SIZE_XXSMALL
    (sw, sh) = cv2.getTextSize(s, font, fontScale=font_size, thickness=1)[0]
    if not min_width is None and sw<min_width: sw = min_width
    x,y = (xy[0], xy[1])
    sx, sy = (x, y-4)
    bg = ((x, y), (x + sw, y - sh - 6))
    cv2.rectangle(image, bg[0], bg[1], bgcolor, cv2.FILLED)
    draw_text(image, s, (sx,sy), font=font, font_size=font_size, thickness=thickness, color=color)
    
def img_resize(img, heigth, width=None):
    if width is None or width<=0:
        width = int((heigth/img.shape[0])*img.shape[1])
    if heigth is None or heigth<=0:
        heigth = int((width/img.shape[1])*img.shape[0])
    img_ = cv2.resize(img, (width, heigth))   
    return img_

def img_histo(img, bw=True, bitmap=False):
    fig, ax = plt.subplots()
    ax.set_title("Histogram")
    ax.set_xlabel("Bins")
    ax.set_ylabel("#pixels")
    if len(img.shape)==3:
        chans=cv2.split(img)
        colors=("b", "g", "r")
        if bw:
            bwimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            colors=("b", "g", "r", 'k')
            chans = (chans[0],chans[1],chans[2],bwimg)
    else:
        colors=('k',)
        chans = (img,)
    for (chan, c) in zip(chans, colors):
        hist=cv2.calcHist([chan], [0], None, [256], [0,256])
        ax.plot(hist, color=c)
        ax.set_xlim([0,256])
    if bitmap:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        himg = np.array(Image.open(buf))
        buf.close()
        return himg
    else:
        plt.show()

def contour_filter(contours, min_area=-1):
    out = []
    for c in contours:
        if min_area>0:
            if cv2.contourArea(c) < min_area: continue
        out.append(c)
    return out

def contour_boxes(contours, wh=False):
    l = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        l.append((x, y, w, h) if wh else (x, y, x+w, y+h))
    return l

def box_filter(bbs, min_area=-1):
    out = []
    for bb in bbs:
        if min_area>0:
            if (bb[0]-bb[2])*(bb[1]-bb[3]) < min_area: continue
        out.append(bb)
    return out

def draw_boxes(img, boxes, color=WHITE, thickness=1, wh=False, labels=False, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=.6, pad=3):
    for i, bb in enumerate(boxes):
        (x1, y1, x2, y2) = bb
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels:
            s = str(i)
            (sw, sh) = cv2.getTextSize(s, font, fontScale=font_size, thickness=thickness)[0]
            xy = (x1+pad, y1+sw+pad)
            cv2.putText(img, str(s), xy, font, font_size, color, thickness, cv2.LINE_AA)

def box_combine(a, b):
    (ax1, ay1, ax2, ay2) = a
    (bx1, by1, bx2, by2) = b
    return (min(ax1,bx1),min(ay1,by1),max(ax2,bx2),max(ay2,by2))

def merge_boxes(l, max_dist=-1):
    m = True
    limit = len(l)**2
    n = 0
    while m:
        skip = {}
        if n>limit: break
        m = False
        l_ = []
        for i in range(0, len(l)):
            a = l[i]
            if i in skip: continue
            for j in range(i+1, len(l)):
                b = l[j]
                merge = False
                if util.geom_util.rintercept(a, b):
                    merge = True
                if max_dist>0 and util.geom_util.rdist2(a, b)<max_dist:    
                     merge = True
                if merge:
                    a_ = box_combine(a, b)
                    #print('!merge:', i, j, a, b, a_)
                    skip[j] = True
                    m = True
                    a = a_
            l_.append(a)
            n += 1
        l = l_
    return l_           
        
        
DEFAULT_TRACKER = 'kcf'

def create_tracker(kind=DEFAULT_TRACKER):
    if kind is None: kind = DEFAULT_TRACKER
    OPENCV_TRACKERS = {
            #"csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            #"boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            #"tld": cv2.TrackerTLD_create,
            #"medianflow": cv2.TrackerMedianFlow_create,
            #"mosse": cv2.TrackerMOSSE_create
        }
    tracker_builder = OPENCV_TRACKERS[kind]
    return tracker_builder()


