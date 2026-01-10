#pip install matplotlib
#pip install opencv-contrib-python
#pip install mahotas
#pip install numpy
#pip install scipy
#pip install scikit-learn

from __future__ import print_function
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mahotas
from os.path import exists
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the img")
args = vars(ap.parse_args())

fn = args["image"]

if not exists(fn):     
    sys.exit("ERROR: File not found: {}".format(fn))  
 

img = cv2.imread(fn)
w = img.shape[1]
h = img.shape[0]
print("Loaded: {} ; w:{} h:{} c:{}".format(fn, w, h, img.shape[2]))


#imutils

def translate(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    image2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return image2

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img2 = cv2.warpAffine(image, M, (w, h))
    return img2

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    img2 = cv2.resize(image, dim, interpolation = inter)
    return img2

def add(image, value):
    M = np.ones(image.shape, dtype = "uint8") * value
    img2 = cv2.add(image, M)
    return img2

def sub(image, value):
    M = np.ones(image.shape, dtype = "uint8") * value
    img2 = cv2.subtract(image, M)
    return img2


img = resize(img, 600)
w = img.shape[1]
h = img.shape[0]

#cv2.imshow("Image:" + fn, img)

#cv2.imwrite("newimage.jpg", img)

#img[0:h//2, 0:w//2] = (0, 255, 0)
#cv2.imshow("Updated", img)

img_ = img[0:h//10, 0:w//10]
#cv2.imshow("Cropped", img_)

canvas = np.zeros((300, 300, 3), dtype = "uint8")
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
white = (255, 255, 255)

cv2.line(canvas, (0, 0), (300, 300), green)
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)

(cx0, cy0) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
for r in range(0, 175, 25):
    cv2.circle(canvas, (cx0, cy0), r, white)

for i in range(0, 25):
    r = np.random.randint(5, high = 200)
    color = np.random.randint(0, high = 256, size = (3,)).tolist()
    p = np.random.randint(0, high = 300, size = (2,))
    cv2.circle(canvas, tuple(p), r, color, -1)

#cv2.imshow("Canvas", canvas)

#img2 = translate(img, 0, 100)
#cv2.imshow("Tx", img2)

#img2 = rotate(img, 180)
#cv2.imshow("Rot(180)", img2)

#img2 = resize(img, 180)
#cv2.imshow("Resize", img2)

#img2 = cv2.flip(img, -1)
#cv2.imshow("Flip", img2)

#img2 = add(img, 100)
#cv2.imshow("Add", img2)

#img2 = sub(img, 50)
#cv2.imshow("Sub", img2)

rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
#cv2.imshow("Rectangle", rectangle)
circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
#cv2.imshow("Circle", circle)

#bitwiseAnd = cv2.bitwise_and(rectangle, circle)
#cv2.imshow("AND", bitwiseAnd)
#bitwiseOr = cv2.bitwise_or(rectangle, circle)
#cv2.imshow("OR", bitwiseOr)
#bitwiseXor = cv2.bitwise_xor(rectangle, circle)
#cv2.imshow("XOR", bitwiseXor)
#bitwiseNot = cv2.bitwise_not(circle)
#cv2.imshow("NOT", bitwiseNot)

mask = np.zeros(img.shape[:2], dtype = "uint8")
(cx, cy) = (w//2, h//2)
cv2.circle(mask, (cx, cy), 100, 255, -1)
masked = cv2.bitwise_and(img, img, mask = mask)
#cv2.imshow("Mask", mask)
#cv2.imshow("Masked", masked)

(B, G, R) = cv2.split(img)
#cv2.imshow("Red", R)
#cv2.imshow("Green", G)
#cv2.imshow("Blue", B)
merged = cv2.merge([B, G, R])
#cv2.imshow("Merged", merged)

zeros = np.zeros(img.shape[:2], dtype = "uint8")
R_ = cv2.merge([zeros, zeros, R])
#cv2.imshow("Red", R_)
G_ = cv2.merge([zeros, G, zeros])
#cv2.imshow("Green", G_)
B_ = cv2.merge([B, zeros, zeros])
#cv2.imshow("Blue", B_)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray", gray)
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.imshow("HSV", hsv)
#lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#cv2.imshow("L*a*b*", lab)


hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#plt.figure()
#plt.title("Grayscale Histogram")
#plt.xlabel("Bins")
#plt.ylabel("# of Pixels")
#plt.plot(hist)
#plt.xlim([0, 256])
#plt.show()


def plot_histogram(image, title, mask = None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
    
#plt.figure()
#plt.title("Color Histogram")
#plt.xlabel("Bins")
#plt.ylabel("# of Pixels")
#for (chan, color) in zip(chans, colors):
#    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#    plt.plot(hist, color = color)
#    plt.xlim([0, 256])
#plt.show()

#plot_histogram(img, "Histogram")

def map_histogram(image, title):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("GB Histogram")
    plt.colorbar(p)
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("GR Histogram")
    plt.colorbar(p)
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("BR Histogram")
    plt.colorbar(p)
    plt.show()
    print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

#eq = cv2.equalizeHist(gray)
#cv2.imshow("Histogram Equalization", np.hstack([gray, eq]))

cv2.imshow("Image:" + fn, img)
cv2.imshow("Gray", gray)

blurred = np.hstack([cv2.blur(img, (3, 3)), cv2.blur(img, (5, 5)), cv2.blur(img, (7, 7))])
#cv2.imshow("Averaged", blurred)
blurred = np.hstack([cv2.GaussianBlur(img, (3, 3), 0), cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(img, (7, 7), 0)])
#cv2.imshow("Gaussian", blurred)
blurred = np.hstack([cv2.medianBlur(img, 3), cv2.medianBlur(img, 5), cv2.medianBlur(img, 7)])
#cv2.imshow("Median", blurred)
blurred = np.hstack([cv2.bilateralFilter(img, 5, 21, 21), cv2.bilateralFilter(img, 7, 31, 31), cv2.bilateralFilter(img, 9, 41, 41)]) 
#cv2.imshow("Bilateral", blurred)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
#cv2.imshow("Threshold Binary", thresh)
#(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("Threshold Binary Inverse", threshInv)
#cv2.imshow("Coins", cv2.bitwise_and(blurred, blurred, mask = threshInv))

thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
#cv2.imshow("Mean Thresh", thresh)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
#cv2.imshow("Gaussian Thresh", thresh)

T = mahotas.thresholding.otsu(blurred)
print("Otsu’s threshold: {}".format(T))
thresh = blurred.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
#cv2.imshow("Otsu", thresh)

T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh = blurred.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
#cv2.imshow("Riddler-Calvard", thresh)

lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
#cv2.imshow("Laplacian", lap)

sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
#cv2.imshow("Sobel X", sobelX)
#cv2.imshow("Sobel Y", sobelY)
#cv2.imshow("Sobel Combined", sobelCombined)

canny = cv2.Canny(blurred, 30, 150)
cv2.imshow("Canny", canny)

(_, cnts) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
#img2 = img.copy()
#cv2.drawContours(img2, cnts, -1, (0, 255, 0), 2)
#cv2.imshow("Countours", img2)

cv2.waitKey(0)
