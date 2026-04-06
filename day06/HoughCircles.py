import cv2 as cv
import numpy as np
from sample_download import get_sample

# 동전 이미지 로드
#img = cv.imread(get_sample('coins.png'))
img = cv.imread(get_sample('coins_connected.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)  # 노이즈 제거

# Hough Circle Transform
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                         param1=50, param2=30,           # Canny 상한값, 누산기 임계값
                         minRadius=15, maxRadius=30)     # 원 크기 범위

# 검출된 원 그리기
result = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # 원 테두리
        cv.circle(result, center, radius, (0, 255, 0), 2)
        # 원 중심점
        cv.circle(result, center, 2, (0, 0, 255), 3)

cv.imshow('Original', img)
cv.imshow('Gray', gray)
cv.imshow('Hough Circles', result)
cv.waitKey(0)
cv.destroyAllWindows()
