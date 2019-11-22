import cv2
import numpy as np
import imutils

# Create some random colors
color = np.random.randint(0,255,(100,3))

croppedFirst = True
first = True
stitched = False
if stitched is True:
    stitchedFirst = True
else:
    stitchedFirst = False

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
    frameSkipped = 5
    prev_frame = frame[:]
    ret, frame = cap.read()
    frameCount += frameSkipped+1
    cap.set(1, frameCount)
    if ret:
        im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #im1 = cv2.GaussianBlur(im1, (21, 21), 0)

        if stitched is True:
            if stitchedFirst is True:
                im2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                stitchedFirst = False
        else:
            im2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #if first is True:
            #firstFrame = im2
            #im2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #im2 = firstFrame
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
        elif featureDetectorType is "HARRIS":
            #not finished
            kp1 = cv2.goodFeaturesToTrack(im1, useHarris=1, mask=None, **feature_params)
            kp2 = cv2.goodFeaturesToTrack(im2, useHarris=1, mask=None, **feature_params)

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
            p0 = kp1
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
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            imMatches = cv2.drawMatchesKnn(im1, kp1, im2, kp2, [good], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        elif matchType is "normal":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
        transformationType = "homography"
        if transformationType is "homography":
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
            h1, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
            height, width = im2.shape
            im1Reg = cv2.warpPerspective(im1, h1, (width, height), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
            im2Reg = cv2.warpPerspective(im2, h, (width, height), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
            #result = cv2.warpPerspective(im1, h1, (im1.shape[1] + im2.shape[1], im2.shape[0]))
            #result = im1.copy()
            #im2RegSt = im1.copy()
            #result[0:im2Reg.shape[0], 0:im2Reg.shape[1]] = result
            #im2Reg[0:im1.shape[0], 0:im1.shape[1]] = im1
        elif transformationType is "euclidian":
            m = cv2.estimateRigidTransform(points2, points1, fullAffine=False)
            height, width = im2.shape
            im1Reg = cv2.warpAffine(im1, m, (width, height))
            im2Reg = cv2.warpAffine(im2, m, (width, height))


        #stiched them together
        if stitched is True:
            im2 = cv2.warpPerspective(im2, h, (im1.shape[1] + im2.shape[1], im1.shape[0]))
            im2[0:im1.shape[0], 0:im1.shape[1]] = im1
            cv2.imshow("stitched", im2)

        #thresh = cv2.threshold(im2Reg, 0, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("threshold", thresh)

        cropped = False
        if cropped is True and croppedFirst is False:
            cv2.imshow("im2", im2Reg)
            thresh = cv2.threshold(im2Reg, 0, 255, cv2.THRESH_BINARY)[1]
            cv2.imshow("threshold", thresh)

            # find all external contours in the threshold image then find
            # the *largest* contour which will be the contour/outline of
            # the stitched image
            im2RegCropped = cv2.copyMakeBorder(im2Reg, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # allocate memory for the mask which will contain the
            # rectangular bounding box of the stitched image region
            mask = np.zeros(thresh.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # create two copies of the mask: one to serve as our actual
            # minimum rectangular region and another to serve as a counter
            # for how many pixels need to be removed to form the minimum
            # rectangular region
            minRect = mask.copy()
            sub = mask.copy()

            # keep looping until there are no non-zero pixels left in the
            # subtracted image
            while cv2.countNonZero(sub) > 0:
                # erode the minimum rectangular mask and then subtract
                # the thresholded image from the minimum rectangular mask
                # so we can count if there are any non-zero pixels left
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)

            # find contours in the minimum rectangular mask and then
            # extract the bounding box (x, y)-coordinates
            cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            # use the bounding box coordinates to extract the our final
            # stitched image
            im2RegCropped = im2RegCropped[y:y + h, x:x + w]
            cv2.imshow("crop", im2RegCropped)
        croppedFirst = False

        # calculate optical flow
        if True:
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
        backgroundSubtraction = "absdiff"
        if backgroundSubtraction is "absdiff":
            im2Diff = cv2.absdiff(im2Reg, im1)
        elif backgroundSubtraction is "KNN":
            backSub = cv2.createBackgroundSubtractorKNN()
            im2Diff = backSub.apply(im1Reg)
            #cv2.imshow('fg', fgmask)

        _, im2Thresh = cv2.threshold(im2Diff, 254, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if False:
            _, im2Mask = cv2.threshold(im2Reg, 0, 255, cv2.THRESH_BINARY)

            # grab the image dimensions
            h = im2Mask.shape[0]
            w = im2Mask.shape[1]

            # loop over the image
            cv2.imshow("imT", im2Thresh)
            for y in range(0, h):
                for x in range(0, w):
                    # threshold the pixel
                    if im2Mask[y, x] > 0 and im2Thresh[y, x] > 0:
                        im2Thresh[y, x] = 255
                    else:
                        im2Thresh[y, x] = 0
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

        #cv2.imshow("matches.jpg", imMatches)

        #cv2.imshow("diff", im2Diff)

        cv2.imshow("threshold", im2Thresh)

        #cv2.imshow("dilate", im2Dilate)

        cv2.imshow("im1", im1)

        #cv2.imshow("im1Reg", im1Reg)

        cv2.imshow("im2Reg", im2Reg)

        #cv2.imshow("result", result)

    else:
        print('Could not read frame')
        cap.release()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()