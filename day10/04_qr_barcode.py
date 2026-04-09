"""
[프로젝트 4] QR Code / Barcode Scanner
- 웹캠으로 QR 코드와 바코드를 실시간 인식

필요한 것:
  Linux:   sudo apt-get install libzbar0 && pip install pyzbar opencv-python
  macOS:   brew install zbar && pip install pyzbar opencv-python
  Windows: pip install pyzbar opencv-python

키 조작:
  q — 종료
  s — 현재 프레임 이미지 저장
  c — 감지 이력 초기화

참고: https://github.com/kairess/qrcode_barcode_detection
"""

import cv2
import webbrowser
from pyzbar import pyzbar

# ── 설정 ──────────────────────────────────────────────────
AUTO_OPEN_URL = False   # True로 바꾸면 URL 감지 시 자동으로 브라우저 열기
SAVE_LOG      = True    # True면 감지된 내용을 scan_log.txt에 저장

# ── 색상 정의 ─────────────────────────────────────────────
COLOR_QR      = (0, 255,   0)   # QR 코드 — 초록
COLOR_BARCODE = (0, 165, 255)   # 바코드  — 주황
COLOR_TEXT    = (255, 255, 255) # 텍스트  — 흰색


def get_color(barcode_type):
    """바코드 종류에 따라 색상 선택"""
    if barcode_type == 'QRCODE':
        return COLOR_QR
    return COLOR_BARCODE


def draw_barcode(frame, barcode, color):
    """감지된 바코드에 테두리와 텍스트를 그린다"""
    # 폴리곤 테두리 (바코드가 기울어진 경우도 정확히 그림)
    pts = barcode.polygon
    if len(pts) == 4:
        pts_array = [(p.x, p.y) for p in pts]
        for i in range(4):
            cv2.line(frame, pts_array[i], pts_array[(i+1) % 4], color, 2)

    # 사각형 테두리 (항상 표시)
    x, y, w, h = barcode.rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # 바코드 내용 + 종류 텍스트
    data  = barcode.data.decode('utf-8')
    label = f'{data}  [{barcode.type}]'
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)
    return data


# ── 메인 ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[오류] 웹캠을 열 수 없습니다.")
    exit()

print("QR/Barcode Scanner 시작")
print("  q: 종료 | s: 화면 저장 | c: 이력 초기화")
print(f"  AUTO_OPEN_URL = {AUTO_OPEN_URL}")

detected_set = set()   # 중복 감지 방지
frame_count  = 0

if SAVE_LOG:
    log_file = open('scan_log.txt', 'a')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    barcodes = pyzbar.decode(frame)

    for barcode in barcodes:
        data  = barcode.data.decode('utf-8')
        btype = barcode.type
        color = get_color(btype)

        draw_barcode(frame, barcode, color)

        # 새로 감지된 코드만 처리
        if data not in detected_set:
            detected_set.add(data)
            print(f"[감지] 타입: {btype:10s} | 내용: {data}")

            if SAVE_LOG:
                log_file.write(f"{btype}\t{data}\n")
                log_file.flush()

            # URL이면 자동으로 브라우저 열기
            if AUTO_OPEN_URL and data.startswith('http'):
                print(f"       → 브라우저 열기: {data}")
                webbrowser.open(data)

    # 감지 건수 표시
    cv2.putText(frame, f'감지 누계: {len(detected_set)}건', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    cv2.imshow('QR / Barcode Scanner', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        import time
        fname = f'scan_{int(time.time())}.png'
        cv2.imwrite(fname, frame)
        print(f"[저장] {fname}")
    elif key == ord('c'):
        detected_set.clear()
        print("[초기화] 감지 이력 삭제")

cap.release()
cv2.destroyAllWindows()
if SAVE_LOG:
    log_file.close()

print(f"\n총 {len(detected_set)}개 코드 감지:")
for item in detected_set:
    print(f"  {item}")
