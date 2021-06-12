import cv2
import numpy as np


def morphology(fgmask):

    close_kernel = np.asarray([
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1],
                            [0,1,1,1,1,1,0],
                            [0,0,1,1,1,0,0],
                            [0,1,1,1,1,1,0],
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1],
                            [0,1,1,1,1,1,0],
                            [0,0,1,1,1,0,0],
                            [0,1,1,1,1,1,0],
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1]],dtype=np.uint8)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, close_kernel)
    # open_kernel = np.ones((3,3),dtype=np.uint8)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, open_kernel)
    # close_kernel = np.ones((5,5),dtype=np.uint8)
    # # opening
    # closing

    return fgmask

cap = cv2.VideoCapture('output.mp4')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_bbox = cv2.VideoWriter('output_bbox.avi', fourcc, 30.0, (w,h))
out_binary = cv2.VideoWriter('output_binary.avi', fourcc, 30.0, (w,h))
out_connected = cv2.VideoWriter('output_connected.avi', fourcc, 30.0, (w,h))

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

min_area_threshold = 30 #pixels
min_width_threshold = 10 #pixels
min_height_threshold = 20 #pixels
min_height_to_width = 1.4

while True:
    ok, frame = cap.read()
    if not ok: break
    fgmask = fgbg.apply(frame, learningRate=-1)
    
    morphed = morphology(fgmask.copy())
    count, classes, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)

    final = np.zeros(frame.shape, dtype=frame.dtype) 
    for k in range (count):
        if k == 0: continue
        
        left, top, width, height, area = stats[k][:5]
        if area < min_area_threshold or width < min_width_threshold or height < min_height_threshold or height/width<min_height_to_width:
            continue
        bottom_center = (left + width//2, top + height)
    
        frame = cv2.rectangle(frame, (left, top), (left+width, top+height), (0,0,255), 1)
        frame = cv2.circle(frame, bottom_center, color=(0,255,0), radius = 4, thickness=2)

        final[classes == k] = (0, 0, 255); # create filtered binary image
        
    
    # frame_show = cv2.resize(frame, (int(frame.shape[1]//1.5), int(frame.shape[0]//1.5)))
    # fgmask_show = cv2.resize(fgmask, (fgmask.shape[1]//2, fgmask.shape[0]//2))
    # final_show = cv2.resize(final, (final.shape[1]//2, final.shape[0]//2))
    # cv2.imshow('Frame', frame_show)
    # cv2.imshow('Mask',fgmask_show)
    # cv2.imshow("Connected", final_show)
    # k = cv2.waitKey(1) & 0xff
    # if k == ord('q'):
    #     break

    out_bbox.write(frame)
    out_binary.write(fgmask)
    out_connected.write(final)
    
cap.release()
cv2.destroyAllWindows()