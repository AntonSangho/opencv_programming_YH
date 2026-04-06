import cv2 as cv
import numpy as np

# 1단계: 이미지 로드
img = cv.imread('road.jpg')
print(f"원본 이미지 크기: {img.shape}")

# 이미지 축소 (처리 속도 개선)
scale = 0.2
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
print(f"축소된 이미지 크기: {img_resized.shape}")

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# 2단계: 블러 처리 (약간의 노이즈 완화)
blurred = cv.GaussianBlur(gray, (5, 5), 1)

# 3단계: Canny 에지 검출 (threshold 높임)
edges = cv.Canny(blurred, 100, 200, apertureSize=3)

# 4단계: 모폴로지 연산 (작은 노이즈 제거)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)  # 끊긴 선 연결
edges = cv.erode(edges, kernel, iterations=1)  # 작은 노이즈 제거

# 5단계: 허프 직선 변환
lines = cv.HoughLinesP(edges, 1, np.pi/180,
                       threshold=50,
                       minLineLength=100,
                       maxLineGap=10)

# 6단계: 검출된 직선을 원본 이미지에 그리기
result = img_resized.copy()
if lines is not None:
    print(f"검출된 직선 수: {len(lines)}")
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Gray', gray)
cv.imshow('Blurred', blurred)
cv.imshow('Edges (모폴로지 적용)', edges)
cv.imshow('Hough Lines', result)
cv.waitKey(0)
cv.destroyAllWindows()
