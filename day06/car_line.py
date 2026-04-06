import cv2 as cv
import numpy as np
#from sample_download import get_sample

# 1단계: 이미지 로드
#img = cv.imread(get_sample('road.jpg'))
#img = cv.imread('road.jpg')
img_resized = cv.imread('road.png')
#print(f"원본 이미지 크기 :{img.shape}") # 실제 크기 확인 

#scale = 0.2 
#img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
print(f"축소된 이미지 크기 :{img_resized.shape}") # 실제 크기 확인 
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# 2단계: Canny 에지 검출 (day04.md 참고)
#edges = cv.Canny(gray, 50, 150, apertureSize=3)
edges = cv.Canny(gray, 100, 200, apertureSize=3)

# 3단계: 허프 직선 변환
#lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

# 4단계: 검출된 직선을 원본 이미지에 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Original', gray)
cv.imshow('Edges', edges)
cv.imshow('Hough Lines', img_resized)
cv.waitKey(0)
cv.destroyAllWindows()
