"""
[프로젝트 1] Annoying Orange Face
- 웹캠 얼굴에서 눈/입을 추출해 오렌지 캐릭터에 합성

필요한 것:
  pip install dlib imutils
  shape_predictor_68_face_landmarks.dat  (아래 명령으로 다운로드)
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

참고: https://github.com/kairess/annoying-orange-face
"""

import cv2
import dlib
import numpy as np
from imutils import face_utils

# ── 1. 초기화 ─────────────────────────────────────────────
orange_img = cv2.imread('orange.jpg')
orange_img = cv2.resize(orange_img, (512, 512))

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

# 오렌지 이미지에서 눈/입이 위치할 중심 좌표 (픽셀 직접 확인 후 설정)
ORANGE_LEFT_EYE  = (200, 180)
ORANGE_RIGHT_EYE = (310, 180)
ORANGE_MOUTH     = (255, 340)


# ── 2. 헬퍼 함수 ──────────────────────────────────────────
def get_roi(img, points):
    """랜드마크 점들을 감싸는 최소 사각형 영역을 잘라낸다"""
    x, y, w, h = cv2.boundingRect(points)
    roi = img[y:y+h, x:x+w]
    return roi, (x, y, w, h)


def clone_to_target(roi, target_img, center):
    """roi를 target_img의 center 위치에 포아송 블렌딩으로 합성"""
    if roi.size == 0:
        return target_img

    h, w = roi.shape[:2]
    # seamlessClone 최소 크기 조건 (너무 작으면 오류)
    if w < 3 or h < 3:
        return target_img

    mask = 255 * np.ones(roi.shape, roi.dtype)
    result = cv2.seamlessClone(roi, target_img, mask, center, cv2.NORMAL_CLONE)
    return result


# ── 3. 메인 루프 ──────────────────────────────────────────
print("웹캠 시작 — 'q'로 종료 | 'l'로 랜드마크 표시 토글")
show_landmarks = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (512, 512))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    result = orange_img.copy()

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # 랜드마크 시각화 (토글)
        if show_landmarks:
            for idx, (x, y) in enumerate(shape):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)

        # 눈/입 ROI 추출 (랜드마크 번호)
        left_eye_roi,  _ = get_roi(frame, shape[36:42])
        right_eye_roi, _ = get_roi(frame, shape[42:48])
        mouth_roi,     _ = get_roi(frame, shape[48:58])

        # 오렌지에 합성
        result = clone_to_target(left_eye_roi,  result, ORANGE_LEFT_EYE)
        result = clone_to_target(right_eye_roi, result, ORANGE_RIGHT_EYE)
        result = clone_to_target(mouth_roi,     result, ORANGE_MOUTH)

    cv2.imshow('Annoying Orange', result)
    if show_landmarks:
        cv2.imshow('Landmarks', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        show_landmarks = not show_landmarks
        if not show_landmarks:
            cv2.destroyWindow('Landmarks')

cap.release()
cv2.destroyAllWindows()
