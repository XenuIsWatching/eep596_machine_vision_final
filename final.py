import cv2
import numpy as np


cap = cv2.VideoCapture('Study clip 057.mpg')
ret, frame = cap.read()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

#res = cv2.CreateMat(frame_height, frame_width, cv2.CV_8U)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

while(cap.isOpened()):
    prev_frame = frame[:]
    ret, frame = cap.read()
    if ret:
        im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        #detect key feature points
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(im1, None)
        kp2, des2 = sift.detectAndCompute(im2, None)

        #some magic with prev_frame


        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append(m)

        #matches = bf.match(des1, des2)
        # Sort matches by score
        #matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        #numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        #matches = matches[:numGoodMatches]

        # Draw top matches
        #imMatches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)

        # Extract location of good matches
        #points1 = np.zeros((len(matches), 2), dtype=np.float32)
        #points2 = np.zeros((len(matches), 2), dtype=np.float32)

        #for i, match in enumerate(matches):
        #    points1[i, :] = kp1[match.queryIdx].pt
        #    points2[i, :] = kp2[match.trainIdx].pt

        points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC,3.0)

        # Use homography
        height, width = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
        im2Reg = cv2.warpPerspective(im2, h, (width, height), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

        imMatches = cv2.drawMatchesKnn(im1, kp1, im2, kp2, [good], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow("matches.jpg", imMatches)

        out.write(im1Reg)

        cv2.imshow("grayframe", im2Reg)

        cv2.absdiff(im2Reg, im2, im2Reg)

        cv2.imshow("absDiff", im2Reg)
    else:
        print('Could not read frame')

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()