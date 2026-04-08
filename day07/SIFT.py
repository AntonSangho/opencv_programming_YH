import numpy as np
import cv2 as cv
from sample_download import get_sample

img = cv.imread(get_sample('home.jpg'))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)
kp,des = sift.compute(gray,kp)


#kp, des = sift.detectAndCompute(gray, None )

img=cv.drawKeypoints(gray,kp,img)

cv.imwrite('sift_keypoint_1.jpg',img)