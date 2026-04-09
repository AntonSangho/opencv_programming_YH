"""
[프로젝트 2] ASCII Art
- 이미지 / 동영상 / 웹캠을 ASCII 문자로 변환

필요한 것:
  pip install opencv-python

MODE 변수를 바꿔서 세 가지 모드를 전환합니다.
  'image'  — 이미지 파일 1장
  'video'  — 동영상 파일
  'webcam' — 웹캠 실시간

참고: https://github.com/kairess/ascii-art
"""

import cv2
import sys

# ── 설정 ──────────────────────────────────────────────────
#MODE       = 'webcam'          # 'image' | 'video' | 'webcam'
MODE       = 'image'          # 'image' | 'video' | 'webcam'
IMAGE_PATH = 'peter.jpg'       # MODE='image' 일 때 사용
VIDEO_PATH = 'imgs/bikini.mp4' # MODE='video' 일 때 사용
COLS       = 100               # 터미널 가로 문자 수

# 밝음 → 어두움 순서의 ASCII 팔레트
CHARS = ' .,-~:;=!*#$@'


# ── 헬퍼 함수 ─────────────────────────────────────────────
def frame_to_ascii(gray, cols=COLS):
    """그레이스케일 이미지를 ASCII 문자열로 변환"""
    h, w = gray.shape
    # 터미널 글자는 세로가 더 길어서 가로를 2배로 설정
    new_w = cols * 2
    new_h = int(h / w * cols * 0.55)
    resized = cv2.resize(gray, (new_w, new_h))

    lines = []
    for row in resized:
        line = ''
        for pixel in row:
            # 픽셀값 0~255 → chars 인덱스 0~(len-1) 변환
            idx = min(int(pixel / 256 * len(CHARS)), len(CHARS) - 1)
            line += CHARS[idx]
        lines.append(line)
    return '\n'.join(lines)


def clear_terminal():
    """터미널 화면 지우기 (애니메이션용)"""
    print('\x1b[2J\x1b[H', end='')


# ── 이미지 모드 ───────────────────────────────────────────
def run_image():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[오류] 이미지를 찾을 수 없습니다: {IMAGE_PATH}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(frame_to_ascii(gray))


# ── 동영상 모드 ───────────────────────────────────────────
def run_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[오류] 동영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    print("동영상 ASCII art 시작 — Ctrl+C로 종료")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 끝나면 처음으로
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clear_terminal()
            print(frame_to_ascii(gray))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


# ── 웹캠 모드 ─────────────────────────────────────────────
def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return

    print("웹캠 ASCII art 시작 — Ctrl+C로 종료")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clear_terminal()
            print(frame_to_ascii(gray))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


# ── 실행 ──────────────────────────────────────────────────
if __name__ == '__main__':
    if MODE == 'image':
        run_image()
    elif MODE == 'video':
        run_video()
    elif MODE == 'webcam':
        run_webcam()
    else:
        print(f"[오류] 알 수 없는 MODE: {MODE}")
        print("MODE를 'image', 'video', 'webcam' 중 하나로 설정하세요.")
        sys.exit(1)
