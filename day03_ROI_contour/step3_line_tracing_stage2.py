import cv2 as cv
import numpy as np

# 웹캠 연결
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

cv.namedWindow('Line Tracing Stage 2', cv.WINDOW_NORMAL)
cv.resizeWindow('Line Tracing Stage 2', 800, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 그레이스케일 + 이진화
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # --- 새로 추가: 노이즈 제거 ---
    # 메디안 필터
    binary = cv.medianBlur(binary, 5)
    
    # 모폴로지 열기 (작은 노이즈 제거)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    
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

        if area > 100:
            valid_count += 1
            print(f" ✓ (조건 만족)", end="")
        print()  # 줄 바꿈

        if area > max_area:
            max_area = area
            largest_cnt = cnt

    # 🔴 디버그 3: 조건 만족하는 contour 개수
    print(f"[조건 만족 (면적 > 100): {valid_count}개]")

    # 분석 및 표시
    if largest_cnt is not None and max_area > 100:
        M = cv.moments(largest_cnt)
        # 🔴 디버그 4: moments 값
        print(f"[moments] m00={M['m00']:.1f}, m10={M['m10']:.1f}, m01={M['m01']:.1f}")

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 🔴 디버그 5: 중심점과 면적
            print(f"[중심점] 좌표: ({cx}, {cy}), 면적: {max_area:.1f}")

            # 컨투어 + 중심점 그리기
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            # --- 새로 추가: 방향 계산 (fitLine) ---
            vx, vy, x, y = cv.fitLine(largest_cnt, cv.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(float(vy[0]), float(vx[0])) * 180 / np.pi

            # 🔴 디버그 6: fitLine 결과
            print(f"[fitLine] vx={vx[0]:.3f}, vy={vy[0]:.3f}, 각도={angle:.1f} deg")

            # --- 새로 추가: 제어신호 생성 ---
            frame_center_x = frame.shape[1] // 2
            error = cx - frame_center_x
            steer = error / frame_center_x

            # 🔴 디버그 7: 제어신호
            print(f"[제어신호] 프레임중앙={frame_center_x}, 중심x={cx}, 오차={error:.1f}, 조향={steer:.2f}")

            # 프레임 중앙 기준선 그리기
            cv.line(frame, (frame_center_x, 0), (frame_center_x, frame.shape[0]),
                   (200, 200, 200), 1)

            # 조향값 시각화 (빨강/파강)
            steer_bar_length = int(steer * 50)
            if steer < 0:
                cv.line(frame, (frame_center_x, frame.shape[0]//2),
                       (frame_center_x + steer_bar_length, frame.shape[0]//2),
                       (0, 0, 255), 3)  # 빨강 (우회전)
            else:
                cv.line(frame, (frame_center_x, frame.shape[0]//2),
                       (frame_center_x + steer_bar_length, frame.shape[0]//2),
                       (255, 0, 0), 3)  # 파강 (좌회전)

            # 정보 표시
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Angle: {angle:.1f} deg', (10, 60),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Steer: {steer:.2f}', (10, 90),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("[⚠️  조건 만족 contour 없음]")
    
    # 결과 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])
    cv.imshow('Line Tracing Stage 2', result)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
