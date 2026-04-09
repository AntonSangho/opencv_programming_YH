"""
[프로젝트 3] Pencil Sketch Animation
- 동영상/웹캠을 연필 스케치 스타일로 변환하고 MP4로 저장

필요한 것:
  pip install opencv-python

MODE 변수를 바꿔서 두 가지 모드를 전환합니다.
  'video'  — 동영상 파일 변환 후 저장
  'webcam' — 웹캠 실시간 (저장 없음)

파라미터 실험:
  sigma_s      — 공간 범위 (20~200), 클수록 굵은 선
  sigma_r      — 색상 범위 (0.01~0.5), 작을수록 엣지 선명
  shade_factor — 음영 강도 (0.005~0.1), 클수록 진한 스케치

참고: https://github.com/kairess/pencil-sketch-animation
"""

import cv2
import numpy as np
import random
import time

# ── 설정 ──────────────────────────────────────────────────
MODE       = 'webcam'        # 'video' | 'webcam'
VIDEO_PATH = 'muyaho.mp4'   # MODE='video' 일 때 사용

# pencilSketch 파라미터 — 여기를 바꿔보세요
SIGMA_S      = 60     # 공간 범위: 20 / 60 / 150 으로 바꿔보기
SIGMA_R      = 0.05   # 색상 범위: 0.01 / 0.05 / 0.3 으로 바꿔보기
SHADE_FACTOR = 0.015  # 음영 강도: 0.005 / 0.015 / 0.05 으로 바꿔보기

# 랜덤 흔들림 효과 설정
SHAKE_PROB     = 0.1   # 흔들림 발생 확률 (0~1)
SHAKE_ANGLE    = 3     # 최대 회전 각도 (도)
SHAKE_PIXELS   = 10    # 최대 이동 픽셀


# ── 헬퍼 함수 ─────────────────────────────────────────────
def apply_shake(frame, width, height):
    """랜덤 회전 + 이동으로 손그림 느낌의 흔들림 추가"""
    angle = random.uniform(-SHAKE_ANGLE, SHAKE_ANGLE)
    tx    = random.randint(-SHAKE_PIXELS, SHAKE_PIXELS)
    ty    = random.randint(-SHAKE_PIXELS, SHAKE_PIXELS)

    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    M[0][2] += tx
    M[1][2] += ty
    return cv2.warpAffine(frame, M, (width, height))


def process_frame(frame, width, height):
    """프레임 한 장을 연필 스케치로 변환"""
    # 10% 확률로 흔들림 적용
    if random.random() < SHAKE_PROB:
        frame = apply_shake(frame, width, height)

    # 잡티 제거용 블러
    frame = cv2.GaussianBlur(frame, (9, 9), 0)

    # 연필 스케치 변환
    gray_sketch, color_sketch = cv2.pencilSketch(
        frame,
        sigma_s=SIGMA_S,
        sigma_r=SIGMA_R,
        shade_factor=SHADE_FACTOR
    )
    return gray_sketch, color_sketch


# ── 동영상 모드 ───────────────────────────────────────────
def run_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 파일 (절반 속도로 저장)
    output_path = f'sketch_{int(time.time())}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / 2, (width, height))

    print(f"동영상 처리 시작: {VIDEO_PATH} → {output_path}")
    print("'q'로 중단, 's'로 현재 프레임 저장")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_sketch, _ = process_frame(frame, width, height)
        out.write(cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR))

        cv2.imshow('Pencil Sketch', gray_sketch)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f'sketch_frame_{int(time.time())}.png'
            cv2.imwrite(fname, gray_sketch)
            print(f"[저장] {fname}")

    cap.release()
    out.release()
    print(f"[완료] {output_path} 저장됨")


# ── 웹캠 모드 ─────────────────────────────────────────────
def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("웹캠 Pencil Sketch 시작")
    print("'q' 종료 | 's' 저장 | 'c' 컬러/흑백 전환")
    show_color = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray_sketch, color_sketch = process_frame(frame, width, height)

        display = color_sketch if show_color else gray_sketch
        cv2.imshow('Pencil Sketch (q:종료 / s:저장 / c:컬러전환)', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f'sketch_{int(time.time())}.png'
            cv2.imwrite(fname, display)
            print(f"[저장] {fname}")
        elif key == ord('c'):
            show_color = not show_color

    cap.release()
    cv2.destroyAllWindows()


# ── 파라미터 비교 실험용 ──────────────────────────────────
def compare_params():
    """sigma_s, shade_factor 값 비교 화면 출력 (단일 이미지)"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[오류] 웹캠 프레임을 읽을 수 없습니다.")
        return

    # sigma_s 비교
    results = []
    for s in [20, 60, 150]:
        g, _ = cv2.pencilSketch(frame, sigma_s=s, sigma_r=0.05, shade_factor=0.015)
        g_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.putText(g_bgr, f'sigma_s={s}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        results.append(g_bgr)

    cv2.imshow('sigma_s 비교 (20 / 60 / 150)', np.hstack(results))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── 실행 ──────────────────────────────────────────────────
if __name__ == '__main__':
    # 파라미터 비교 실험: compare_params() 주석 해제 후 실행
    # compare_params()

    if MODE == 'video':
        run_video()
    elif MODE == 'webcam':
        run_webcam()
