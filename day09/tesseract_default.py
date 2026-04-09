import cv2
import pytesseract
import numpy as np

# 이미지 로드
img = cv2.imread('경기부천아7683.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 전처리 파이프라인
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(gray)

binary = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# 전처리된 이미지로 OCR
text = pytesseract.image_to_string(binary, lang='kor+eng')
print(f"인식결과 : {text}")

# 상세 정보 ( 신뢰도 포함 )
data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
print(f"신뢰도 : {data['conf']}") # 각 글자의 신뢰도 