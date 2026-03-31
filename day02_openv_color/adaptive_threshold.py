import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
import urllib.request
import os

# github sample 주소 : https://github.com/opencv/opencv/tree/master/samples/data
def get_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename)

# 파일 읽기
img = get_sample("sudoku.png")
img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

img = cv.medianBlur(img,5)

ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

#rest,thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#rest,thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
#rest,thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
#rest,thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
#rest,thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,3,i+1),plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()