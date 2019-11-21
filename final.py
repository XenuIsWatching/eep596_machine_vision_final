import cv2
import numpy as np
import imutils

# Create some random colors
color = np.random.randint(0,255,(100,3))

first = True

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture('Study clip 057.mpg')
ret, frame = cap.read()
frameCount = 0

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

while(cap.isOpened()):
    frameSkipped = 1
    prev_frame = frame[:]
    ret, frame = cap.read()
    frameCount += frameSkipped+1
    cap.set(1, frameCount)
    if ret:
        im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #im1 = cv2.GaussianBlur(im1, (21, 21), 0)
        im2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        if first is True:
            firstFrame = im2
        im2 = firstFrame
        #im2 = cv2.GaussianBlur(im2, (21, 21), 0)

        #im1 = cv2.equalizeHist(im1)
        #im2 = cv2.equalizeHist(im1)

        #detect key feature points
        featureDetectorType = "ORB"
        if featureDetectorType is "SIFT":
            detector = cv2.xfeatures2d.SIFT_create()
            kp1 = detector.detect(im1)
            kp2 = detector.detect(im2)
        elif featureDetectorType is "SURF":
            detector = cv2.xfeatures2d.SURF_create()
            kp1 = detector.detect(im1)
            kp2 = detector.detect(im2)
        elif featureDetectorType is "ORB":
            detector = cv2.ORB_create(nfeatures=1500)
            kp1 = detector.detect(im1)
            kp2 = detector.detect(im2)

        featureDescriptorType = "ORB"
        if featureDescriptorType is "SIFT":
            descriptor = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = detector.compute(im1, kp1)
            kp2, des2 = detector.compute(im2, kp2)
        elif featureDescriptorType is "SURF":
            descriptor = cv2.xfeatures2d.SURF_create()
            kp1, des1 = detector.compute(im1, kp1)
            kp2, des2 = detector.compute(im2, kp2)
        elif featureDescriptorType is "ORB":
            descriptor = cv2.ORB_create(nfeatures=1500)
            kp1, des1 = detector.compute(im1, kp1)
            kp2, des2 = detector.compute(im2, kp2)

        if first is True:
            p0 = cv2.goodFeaturesToTrack(im2, mask = None, **feature_params)
            # Create a mask image for drawing purposes
            flowMask = np.zeros_like(im2)
            first = False

        #some magic with prev_frame

        # BFMatcher with default params
        matchType = "knn"
        if matchType is "knn":
            bf = cv2.BFMatcher()
            matchesPrevToCurr = bf.knnMatch(des2, des1, k=2)
            matchesCurrToPrev = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            for m, n in matchesCurrToPrev:
                if m.distance < 0.5 * n.distance:
                    good.append(m)

            points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            imMatches = cv2.drawMatchesKnn(im1, kp1, im2, kp2, [good], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        elif matchType is "normal":
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            # Sort matches by score
            matches.sort(key=lambda x: x.distance, reverse=False)

            # Remove not so good matches
            numGoodMatches = int(len(matches) * 0.15)
            matches = matches[:numGoodMatches]

            # Extract location of good matches
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt

            # Draw top matches
            imMatches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)


        # Find homography
        transformationType = "euclidian"
        if transformationType is "homography":
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
            height, width = im2.shape
            im1Reg = cv2.warpPerspective(im1, h, (width, height), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
            im2Reg = cv2.warpPerspective(im2, h, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        elif transformationType is "euclidian":
            m = cv2.estimateRigidTransform(points2, points1, fullAffine=False)
            height, width = im2.shape
            im1Reg = cv2.warpAffine(im1, m, (width, height))
            im2Reg = cv2.warpAffine(im2, m, (width, height))


        # calculate optical flow
        if False:
            p1, st, err = cv2.calcOpticalFlowPyrLK(im2Reg, im1, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                flowMask = cv2.line(flowMask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            flow = cv2.add(im2, flowMask)

            cv2.imshow('flow', flow)

            # Now update the previous frame and previous points
            p0 = good_new.reshape(-1, 1, 2)

        #im2Reg = cv2.GaussianBlur(im2Reg, (21, 21), 0)
        #im1 = cv2.GaussianBlur(im1, (21, 21), 0)

        #remove background
        im2Diff = cv2.absdiff(im2Reg, im1)

        _, im2Thresh = cv2.threshold(im2Diff, 25, 255, cv2.THRESH_BINARY)
        #im2Thresh = cv2.adaptiveThreshold(im2Diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

        #define dilate to fo fill in holes
        im2Dilate = cv2.dilate(im2Thresh, None, iterations=2)

        contours = cv2.findContours(im2Thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(im2Dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 80:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow("new img", new_img)

        cv2.imshow("matches.jpg", imMatches)

        cv2.imshow("im1Reg", im1Reg)

        cv2.imshow("im2Reg", im2Reg)

        cv2.imshow("diff", im2Diff)

        cv2.imshow("threshold", im2Thresh)

        cv2.imshow("dilate", im2Dilate)

        cv2.imshow("blarg", im1)

    else:
        print('Could not read frame')
        cap.release()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()