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

ret1,th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

blur = cv.GaussianBlur(img, (5,5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

images = [img, 0, th1,
          img,  0, th2, 
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding(v=127)', 
          'Origianl Noisy Image', 'Histogra', "Otu's Thresolding",
          'Gaussian filtered Image', 'Histogram', "Otu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
