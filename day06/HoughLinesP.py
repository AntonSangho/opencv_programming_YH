import cv2 as cv
import numpy as np
from sample_download import get_sample

img = cv.imread(get_sample('sudoku.png'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# 선분 방식: (x1, y1, x2, y2) 시작점과 끝점으로 표현
lines = cv.HoughLinesP(edges, 1, np.pi/180, 
                       threshold=50, minLineLength=50, maxLineGap=10)

# 검출된 직선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Hough Lines P', img)
cv.waitKey(0)
cv.destroyAllWindows()
