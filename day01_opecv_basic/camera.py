import numpy as np
import cv2 as cv

frame_count = 0

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    #cv.imshow('frame', gray)
    #frame = cv.flip(frame, -1) 
    cv.imshow('frame',frame)
    key = cv.waitKey(1)
    # c를 누르면 캡쳐가되서 사진을 저장한다.
    #if cv.waitKey(1) == ord('q'):
    if key == ord('q'):
        break
    #if cv.waitKey(1) == ord('c'):
    elif key == ord('c'):
        filename = f"./capture_{frame_count}.png"
        cv.imwrite(filename, frame)
        print(f"캡쳐 저장: {filename}")
        frame_count +=1

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()