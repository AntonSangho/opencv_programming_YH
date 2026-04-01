import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# 웹캠 연결
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Figure와 Axes 생성
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(bottom=0.25)

# 슬라이더용 Axes
ax_block = plt.axes([0.2, 0.15, 0.6, 0.03])
ax_c = plt.axes([0.2, 0.05, 0.6, 0.03])

# 슬라이더 생성
slider_block = Slider(ax_block, 'BlockSize', 3, 31, valinit=11, valstep=2)
slider_c = Slider(ax_c, 'C', 0, 20, valinit=2, valstep=1)

# 이미지 객체 저장 (업데이트용)
ims = [None, None, None, None]

def animate(frame):
    ret, frame_img = cap.read()
    if not ret:
        cap.release()
        plt.close()
        return ims

    # 그레이스케일 변환
    gray = cv.cvtColor(frame_img, cv.COLOR_BGR2GRAY)

    # 노이즈 감소
    gray = cv.medianBlur(gray, 5)

    # 트랙바 값
    block = int(slider_block.val)
    c = int(slider_c.val)

    # 홀수 보장
    if block % 2 == 0:
        block += 1

    # 이진화 4종
    _, global_th = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, otsu_th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mean_th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY, block, c)
    gaussian_th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
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

    return ims

# 애니메이션 생성
ani = FuncAnimation(fig, animate, interval=50, blit=False)

plt.show()

cap.release()
