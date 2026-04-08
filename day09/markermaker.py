import cv2
from cv2 import aruco

# ArUco 딕셔너리 생성
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# ID = 42인 마커 이미지 생성 (200x200 픽셀)
marker_img = aruco.generateImageMarker(aruco_dict, 42, 200)

# 흰색 여백 추가 (인쇄/검출용 — 여백 없으면 카메라 인식 실패)
border = 40
marker_with_border = cv2.copyMakeBorder(
    marker_img, border, border, border, border,
    cv2.BORDER_CONSTANT, value=255
)

# 이미지로 저장 (인쇄용)
cv2.imwrite('aruco_marker_42.jpg', marker_with_border)
print("저장 완료: aruco_marker_42.jpg (인쇄 후 카메라에 사용)")

# 화면에 표시
cv2.imshow('ArUco Marker ID=42', marker_with_border)

cv2.waitKey(0)


