import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 이미지 읽기 (그레이스케일)
img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)

# Figure와 Axes 생성
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(bottom=0.25)

# 슬라이더용 Axes
ax_block = plt.axes([0.2, 0.15, 0.6, 0.03])
ax_c = plt.axes([0.2, 0.05, 0.6, 0.03])

# 슬라이더 생성
slider_block = Slider(ax_block, 'BlockSize', 3, 31, valinit=11, valstep=2)
slider_c = Slider(ax_c, 'C', 0, 20, valinit=2, valstep=1)

def update(val):
    block = int(slider_block.val)
    c = int(slider_c.val)

    # blockSize는 홀수여야 함
    if block % 2 == 0:
        block += 1

    # Global Threshold
    _, global_th = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # Otsu
    _, otsu_th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Adaptive Mean
    mean_th = cv.adaptiveThreshold(img, 255,
                                   cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY, block, c)

    # Adaptive Gaussian
    gaussian_th = cv.adaptiveThreshold(img, 255,
                                       cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, block, c)

    # 화면 업데이트
    axes[0, 0].clear()
    axes[0, 0].imshow(global_th, cmap='gray')
    axes[0, 0].set_title('Global Threshold')
    axes[0, 0].axis('off')

    axes[0, 1].clear()
    axes[0, 1].imshow(otsu_th, cmap='gray')
    axes[0, 1].set_title('Otsu')
    axes[0, 1].axis('off')

    axes[1, 0].clear()
    axes[1, 0].imshow(mean_th, cmap='gray')
    axes[1, 0].set_title(f'Adaptive Mean (block={block}, C={c})')
    axes[1, 0].axis('off')

    axes[1, 1].clear()
    axes[1, 1].imshow(gaussian_th, cmap='gray')
    axes[1, 1].set_title(f'Adaptive Gaussian (block={block}, C={c})')
    axes[1, 1].axis('off')

    fig.canvas.draw_idle()

# 슬라이더 콜백 연결
slider_block.on_changed(update)
slider_c.on_changed(update)

# 초기 표시
update(None)

plt.show()
