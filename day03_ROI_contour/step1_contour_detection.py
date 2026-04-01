import cv2 as cv
import numpy as np

# 웹캠 연결
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

# 창 생성
cv.namedWindow('Line Tracing Stage 1', cv.WINDOW_NORMAL)

# 반복문: 웹캠 프레임 처리
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 이진화 (Otsu 사용)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 컨투어 검출
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 🔴 디버그 1: 검출된 전체 contour 개수
    print(f"\n[프레임 분석] 검출된 contour 개수: {len(contours)}")

    # 가장 큰 컨투어 찾기
    largest_cnt = None
    max_area = 0
    valid_count = 0  # 조건 만족하는 contour 개수

    for idx, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        # 🔴 디버그 2: 각 contour의 면적
        print(f"  contour[{idx}] 면적: {area:.1f}", end="")

        if area > 50:
            valid_count += 1
            print(f" ✓ (조건 만족)", end="")
        print()  # 줄 바꿈

        if area > max_area:
            max_area = area
            largest_cnt = cnt

    # 🔴 디버그 3: 조건 만족하는 contour 개수
    print(f"[조건 만족 (면적 > 50): {valid_count}개]")

    # 중심좌표 계산 (최소 크기 필터) - 루프 밖으로 이동!
    if largest_cnt is not None and max_area > 50:
        M = cv.moments(largest_cnt)
        # 🔴 디버그 4: moments 값
        print(f"[moments] m00={M['m00']:.1f}, m10={M['m10']:.1f}, m01={M['m01']:.1f}")

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 🔴 디버그 5: 중심점과 면적
            print(f"[빨간점 그림] 중심: ({cx}, {cy}), 면적: {max_area:.1f}")

            # 컨투어 그리기
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)

            # 중심점 그리기 (빨강)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            # 좌표 표시
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Area: {max_area:.0f}', (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("[⚠️  조건 만족 contour 없음 - 빨간점 미표시]")

    # 이진화 이미지 + 원본 나란히 표시 - 루프 밖으로 이동!
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    cv.imshow('Line Tracing Stage 1', result)

    # 'q' 키로 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv.destroyAllWindows()
