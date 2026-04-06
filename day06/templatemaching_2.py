import urllib.request
import os
import cv2 as cv 
from matplotlib import pyplot as plt 

def get_sample(filename, repo='opencv'):
    """부교 또는 OpenCV 공식 샘플 이미지 자동 다운로드
    
    Args:
        filename (str): 이미지 파일명 (예: 'morphological.png')
        repo (str): 'insightbook' (부교) 또는 'opencv' (공식)
    
    Returns:
        str: 다운로드된 파일명
    """
    if not os.path.exists(filename):
        if repo == 'insightbook':
            url = f"https://raw.githubusercontent.com/dltpdn/insightbook.opencv_project_python/master/img/{filename}"
        else:  # opencv 공식
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용 방법
# img = cv.imread(get_sample('morphological.png', repo='insightbook'))

# 이미지와 템플릿 로드
img = cv.imread(get_sample('messi5.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

template = cv.imread('template.jpg')
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

# Template Matching — TM_CCOEFF_NORMED (조명 변화에 강함)
result = cv.matchTemplate(gray, template_gray, cv.TM_CCOEFF_NORMED)

# 최적 매칭 위치 찾기
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
top_left = max_loc  # TM_CCOEFF_NORMED에서는 max_loc이 최적

# 매칭 영역 표시
h, w = template_gray.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)

result_img = img.copy()
cv.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

# 화면 표시
cv.imshow('Template', cv.cvtColor(template_gray, cv.COLOR_GRAY2BGR))
cv.imshow('Result', result_img)
cv.imshow('Result Map', result)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Max similarity: {max_val:.4f}")
print(f"Best match location: {top_left}")
