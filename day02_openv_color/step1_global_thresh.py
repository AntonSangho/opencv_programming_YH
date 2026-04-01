import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 이미지 읽기 (그레이스케일)
img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)

# Figure와 Axes 생성
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.35)

# 슬라이더용 Axes
ax_thresh = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_mode = plt.axes([0.2, 0.1, 0.6, 0.03])

# 슬라이더 생성
slider_thresh = Slider(ax_thresh, 'Threshold', 0, 255, valinit=127, valstep=1)
slider_mode = Slider(ax_mode, 'Mode (0: Binary, 1: Inv)', 0, 1, valinit=0, valstep=1)

def update(val):
    thresh = int(slider_thresh.val)
    mode = int(slider_mode.val)

    # 이진화 타입 선택
    if mode == 0:
        thresh_type = cv.THRESH_BINARY
        mode_name = 'BINARY'
    else:
        thresh_type = cv.THRESH_BINARY_INV
        mode_name = 'BINARY_INV'

    # 이진화 적용
    _, result = cv.threshold(img, thresh, 255, thresh_type)

    # 화면 업데이트
    ax.clear()
    ax.imshow(result, cmap='gray')
    ax.set_title(f'Threshold: {thresh} | Mode: {mode_name}')
    ax.axis('off')
    fig.canvas.draw_idle()

# 슬라이더 콜백 연결
slider_thresh.on_changed(update)
slider_mode.on_changed(update)

# 초기 표시
update(None)

plt.show()
