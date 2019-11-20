import cv2
import numpy as np

cap = cv2.VideoCapture('Study clip 017.mpg')
ret, frame = cap.read()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

while(cap.isOpened()):
    prev_frame = frame[:]
    ret, frame = cap.read()
    if ret:
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        #detect key feature points
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        #some magic with prev_frame


        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:
                good.append([m])

        #draw key points detected
        img = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        out.write(img)

        cv2.imshow("grayframe",img)
    else:
        print('Could not read frame')

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()