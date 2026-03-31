import numpy as np
import cv2 as cv

def nothing(x):
    pass

img = np.zeros((300, 512,3), np.uint8)
#cv.namedWindow('image')
cv.namedWindow('Trackbar')

#cv.createTrackbar('R', 'image', 0, 255, nothing)
#cv.createTrackbar('G', 'image', 0, 255, nothing)
#cv.createTrackbar('B', 'image', 0, 255, nothing)
cv.createTrackbar('H_min', 'Trackbar', 35, 179, nothing)
cv.createTrackbar('H_max', 'Trackbar', 85, 179, nothing)
cv.createTrackbar('S_min', 'Trackbar', 50, 255, nothing)
cv.createTrackbar('S_max', 'Trackbar', 255, 255, nothing)
cv.createTrackbar('V_min', 'Trackbar', 50, 255, nothing)
cv.createTrackbar('V_max', 'Trackbar', 255, 255, nothing)


#switch = '0 : OFF \n1 : ON'
#cv.createTrackbar(switch, 'image', 0, 1, nothing)

cap = cv.VideoCapture(0)

while(1):
    #cv.imshow('image', img)
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #k = cv.waitKey(1) & 0xFF
    #if k == 27:
        #break
    
    #r = cv.getTrackbarPos('R', 'image')
    #g = cv.getTrackbarPos('G', 'image')
    #b = cv.getTrackbarPos('B', 'image')
    #s = cv.getTrackbarPos(switch, 'image')

    h_min = cv.getTrackbarPos('H_min', 'Trackbar')
    h_max = cv.getTrackbarPos('H_max', 'Trackbar')
    s_min = cv.getTrackbarPos('S_min', 'Trackbar')
    s_max = cv.getTrackbarPos('S_max', 'Trackbar')
    v_min = cv.getTrackbarPos('V_min', 'Trackbar')
    v_max = cv.getTrackbarPos('V_max', 'Trackbar')

    print(f"H: {h_min} - { h_max}, S: {s_min} - s{s_max}, V: {v_min}")


    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('Original', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Result', result)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

    #if s == 0:
        #img[:] = 0
    #else :
        #img[:] = [b,g,r]