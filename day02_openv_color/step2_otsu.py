import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 이미지 읽기 (그레이스케일)
img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)

# Figure와 Axes 생성
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.25)

# 슬라이더용 Axes
ax_thresh = plt.axes([0.2, 0.1, 0.6, 0.03])

# 슬라이더 생성
slider_thresh = Slider(ax_thresh, 'Manual Threshold', 0, 255, valinit=127, valstep=1)

def update(val):
    manual = int(slider_thresh.val)

    # 수동 이진화
    _, manual_th = cv.threshold(img, manual, 255, cv.THRESH_BINARY)

    # Otsu 자동 이진화
    ret_otsu, otsu_th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 화면 업데이트
    axes[0].clear()
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].clear()
    axes[1].imshow(manual_th, cmap='gray')
    axes[1].set_title(f'Manual Threshold: {manual}')
    axes[1].axis('off')

    axes[2].clear()
    axes[2].imshow(otsu_th, cmap='gray')
    axes[2].set_title(f'Otsu Threshold: {ret_otsu:.0f}')
    axes[2].axis('off')

    fig.canvas.draw_idle()

# 슬라이더 콜백 연결
slider_thresh.on_changed(update)

# 초기 표시
update(None)

plt.show()
