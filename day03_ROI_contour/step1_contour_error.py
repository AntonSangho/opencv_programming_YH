import cv2 as cv
import numpy as np

# 웹캠 연결
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

cv.namedWindow('Line Tracing Stage 1', cv.WINDOW_NORMAL)
cv.resizeWindow('Line Tracing Stage 1', 800, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다")
        break
    
    # 그레이스케일 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 이진화 (Otsu 알고리즘 사용)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # 컨투어 검출
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 컨투어 선택
    largest_cnt = None
    max_area = 0
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt
    
    # 중심좌표 계산 및 표시
    if largest_cnt is not None and max_area > 100:
        M = cv.moments(largest_cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 컨투어 그리기 (녹색)
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            
            # 중심점 그리기 (빨강, 큰 원)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            
            # 좌표와 면적 표시
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Area: {max_area:.0f}', (10, 60),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 이진화 이미지 + 원본 나란히 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    
    cv.imshow('Line Tracing Stage 1', result)
    
    # 'q' 키로 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv.destroyAllWindows()
