import cv2
import numpy as np

cap = cv2.VideoCapture('input.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('Mask',fgmask)
    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1) & 0xff
    
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()