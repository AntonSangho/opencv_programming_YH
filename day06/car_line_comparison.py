import cv2 as cv
import numpy as np

# 이미지 로드
img = cv.imread('road.jpg')
print(f"원본 이미지 크기: {img.shape}")

# 이미지 축소 (처리 속도 개선)
scale = 0.2
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
print(f"축소된 이미지 크기: {img_resized.shape}")

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
h, w = img_resized.shape[:2]

print("\n" + "="*60)
print("4가지 방법 비교")
print("="*60)

# ============ V1: 기본 (Canny 50, 150) ============
edges_v1 = cv.Canny(gray, 50, 150, apertureSize=3)
lines_v1 = cv.HoughLinesP(edges_v1, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
result_v1 = img_resized.copy()
count_v1 = 0
if lines_v1 is not None:
    count_v1 = len(lines_v1)
    for line in lines_v1:
        x1, y1, x2, y2 = line[0]
        cv.line(result_v1, (x1, y1), (x2, y2), (0, 255, 0), 2)
print(f"\n[V1] 기본 (Canny 50, 150) → 검출선: {count_v1}개")

# ============ V2: Canny threshold 높임 (100, 200) ============
edges_v2 = cv.Canny(gray, 100, 200, apertureSize=3)
lines_v2 = cv.HoughLinesP(edges_v2, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
result_v2 = img_resized.copy()
count_v2 = 0
if lines_v2 is not None:
    count_v2 = len(lines_v2)
    for line in lines_v2:
        x1, y1, x2, y2 = line[0]
        cv.line(result_v2, (x1, y1), (x2, y2), (0, 255, 0), 2)
print(f"[V2] Canny 100, 200 → 검출선: {count_v2}개")

# ============ V3: 모폴로지 적용 ============
blurred = cv.GaussianBlur(gray, (5, 5), 1)
edges_v3 = cv.Canny(blurred, 100, 200, apertureSize=3)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
edges_v3 = cv.morphologyEx(edges_v3, cv.MORPH_CLOSE, kernel, iterations=1)
edges_v3 = cv.erode(edges_v3, kernel, iterations=1)
lines_v3 = cv.HoughLinesP(edges_v3, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
result_v3 = img_resized.copy()
count_v3 = 0
if lines_v3 is not None:
    count_v3 = len(lines_v3)
    for line in lines_v3:
        x1, y1, x2, y2 = line[0]
        cv.line(result_v3, (x1, y1), (x2, y2), (0, 255, 0), 2)
print(f"[V3] 모폴로지 적용 → 검출선: {count_v3}개")

# ============ V4: ROI 적용 ============
roi_start = int(h * 0.6)
roi = gray[roi_start:, :]
edges_v4 = cv.Canny(roi, 100, 200, apertureSize=3)
lines_v4 = cv.HoughLinesP(edges_v4, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
result_v4 = img_resized.copy()
count_v4 = 0
if lines_v4 is not None:
    count_v4 = len(lines_v4)
    for line in lines_v4:
        x1, y1, x2, y2 = line[0]
        y1 += roi_start
        y2 += roi_start
        cv.line(result_v4, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.line(result_v4, (0, roi_start), (w, roi_start), (255, 0, 0), 1)
print(f"[V4] ROI 적용 → 검출선: {count_v4}개")

print("\n" + "="*60)
print("결과 분석:")
print("="*60)
print(f"V1 (기본): {count_v1}개 - 잡음이 많음")
print(f"V2 (Canny 높임): {count_v2}개 - 개선됨")
print(f"V3 (모폴로지): {count_v3}개 - 더 정제됨")
print(f"V4 (ROI): {count_v4}개 - 관심 영역만 처리")
print("="*60)

# 결과 표시
cv.imshow('V1 - Basic (Canny 50,150)', result_v1)
cv.imshow('V2 - Canny Higher (100,200)', result_v2)
cv.imshow('V3 - Morphology', result_v3)
cv.imshow('V4 - ROI', result_v4)

print("\n✅ 각 방법을 비교하고 'q'를 눌러 종료하세요")
cv.waitKey(0)
cv.destroyAllWindows()
