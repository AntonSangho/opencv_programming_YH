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
h, w = img_resized.shape[:2]

# 2단계: ROI 설정 (도로 아래쪽만 처리)
roi_start = int(h * 0.6)  # 상단 60% 제외, 하단 40%만 처리
roi = gray[roi_start:, :]

print(f"ROI 크기: {roi.shape}")

# 3단계: Canny 에지 검출
edges = cv.Canny(roi, 100, 200, apertureSize=3)

# 4단계: 허프 직선 변환
lines = cv.HoughLinesP(edges, 1, np.pi/180,
                       threshold=50,
                       minLineLength=100,
                       maxLineGap=10)

# 5단계: 검출된 직선을 원본 이미지에 그리기 (ROI 오프셋 적용)
result = img_resized.copy()
if lines is not None:
    print(f"검출된 직선 수: {len(lines)}")
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # ROI 좌표 → 원본 좌표 변환
        y1 += roi_start
        y2 += roi_start
        cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ROI 영역 표시 (시각화)
cv.line(result, (0, roi_start), (w, roi_start), (255, 0, 0), 1)

cv.imshow('Gray', gray)
cv.imshow('ROI Edges (하단 40%)', edges)
cv.imshow('Hough Lines (파란 선 = ROI 경계)', result)
cv.waitKey(0)
cv.destroyAllWindows()
