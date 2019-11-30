import cv2
import numpy as np
import imutils
from scipy import signal
import math
from matplotlib import pyplot as plt
import py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pdb
import tqdm
import torch.optim as optim
import os
import PIL as pil
import warnings
from scipy.spatial import distance as dist
from collections import OrderedDict
from scipy.spatial import cKDTree

uid = 0

# the cnn class which inherit from torch.nn.Module class
layer = 2

#init data loader
DATADIR = os.getcwd()+'/data/train_img'
BATCH_SIZE = 16
IMG_SIZE = 100
CENTER_SIZE = IMG_SIZE+IMG_SIZE*0.2#20
CATEGORY_SIZE = 2 #how many folders/categories we have in data folder, for now only car, plane, and person

#CATAGORIES = ["car","person","plane"]
CATAGORIES = ["car","motorbike","person","plane"]

# transform to do random affine and cast image to PyTorch tensor
trans_ = torchvision.transforms.Compose(
    [
     # torchvision.transforms.RandomAffine(10),
     torchvision.transforms.Resize((IMG_SIZE)),
     torchvision.transforms.CenterCrop(CENTER_SIZE),
     torchvision.transforms.ToTensor()] #transform from height*width*channel to ch*h*w in order to fit tourch tensor format
)


class CNN(nn.Module):
    cur_kernel_size = 3
    pool_kernel_val = 2
    cur_img_dim = CENTER_SIZE

    def __init__(self):  # constructor
        super(CNN, self).__init__()

        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16)  # 1st convolve layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # down sampling layer
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)  # 2nd convolve layer

        # calculate the final dimention (h*w*d) after 2 layers of convolution and downsampling
        global cur_img_dim
        global cur_kernel_size
        global pool_kernel_val
        cur_img_dim = CENTER_SIZE
        cur_kernel_size = 3
        pool_kernel_val = 2
        img_trim = ((cur_kernel_size - 1) / 2) * 2  # here assume kernel size is always odd
        for i in range(layer):
            cur_img_dim -= img_trim
            cur_img_dim = cur_img_dim / pool_kernel_val
        cur_img_dim = int(cur_img_dim)

        # FC layer (fully-connected or linear layer)
        self.fc1 = nn.Linear(int(32 * cur_img_dim * cur_img_dim), CATEGORY_SIZE)  # 32 * 28 * 28 for 2 layers

    def forward(self, x):
        # define the data flow through the deep learning layers
        # (1st conv->pool layer)
        x = self.pool(F.relu(self.l1(x)))
        # (2nd conv->pool layer)
        x = self.pool(F.relu(self.l2(x)))
        # flatten layer, set -1 coz last batch might not be full
        input_size = 32 * cur_img_dim * cur_img_dim
        x = x.reshape(-1, input_size)  # [16 x 1152]
        # FC layer
        x = self.fc1(x)
        return x


m = CNN()
m = torch.load("model.pt")

cap = cv2.VideoCapture('Data_backup/sample_video/V3V100007_017.avi')
frameSkipped = 1
filterType = "bilateral"
method = "optical"
flowType = "LK"
contourAreaCutoff = 800
lk_params = dict( winSize =(19, 19),
                  maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners=200,
                       qualityLevel=0.001,
                       minDistance=4,
                       blockSize=19)

def image_loader(loader, image_name):
    image = pil.Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def objTypeByPath(img_dir):
    idx = np.argmax(m(image_loader(trans_, img_dir)).detach().numpy())
    return CATAGORIES[idx]

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

def optical_flow(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t,
                                                                                          boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]
    return (u, v)

def optical_flow_sparse(I1g, I2g, window_size, p0, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t,
                                                                                          boundary='symm', mode=mode)
    u = []
    v = []
    # within window window_size * window_size
    for (i, j) in p0[:, 0]:
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]
    return (u, v)

def display_flow(img, flow, stride=1000):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)
        for y in range(0, flow.shape[0] - 1, 15):
            for x in range(0, flow.shape[1] - 1, 15):
                pt1 = (x, y)
                pt2 = (int(x + flow[y, x, 0] / 2), int(y + flow[y, x, 1] / 2))

                if (abs(flow[y, x, 0]) < 5 and abs(flow[y, x, 1]) < 5):
                    img = cv2.arrowedLine(img, pt1, pt2, (255, 255, 255), 1, tipLength=0.4)
                elif (flow[y, x, 0] < 0 and abs(flow[y, x, 1]) <= abs(flow[y, x, 0])):
                    img = cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 2, tipLength=0.4)
                elif (flow[y, x, 0] > 0 and abs(flow[y, x, 1]) <= abs(flow[y, x, 0])):
                    img = cv2.arrowedLine(img, pt1, pt2, (0, 255, 0), 2, tipLength=0.4)
                elif (flow[y, x, 1] < 0):
                    img = cv2.arrowedLine(img, pt1, pt2, (0, 255, 255), 2, tipLength=0.4)
                elif (flow[y, x, 1] > 0):
                    img = cv2.arrowedLine(img, pt1, pt2, (255, 100, 100), 2, tipLength=0.4)
                else:
                    img = cv2.arrowedLine(img, pt1, pt2, (255, 255, 255), 1, tipLength=0.4)
        norm_opt_flow = np.linalg.norm(flow, axis=2)
        normal_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow("optical flow", img)
        cv2.imshow("optical flow magntiude", norm_opt_flow)
        k = cv2.waitKey(1)
        if k is 27:
            return 1
        else:
            return 0

def draw_flow(img, flow, step=16):
    """
    Taken from: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    :param img: Frame to draw the flow
    :param flow: Flow vectors
    :param step: Number of pixels each vector represents
    :return: visualisation
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def calculate_region_of_interest(frame, tracking_points, range):
    mask = np.zeros_like(frame)
    mask[:] = 255
    for(x, y) in tracking_points:
        cv2.circle(mask, (x, y), range, 0, -1)
    cv2.imshow('point mask', mask)
    return mask

# Create some random colors
color = np.random.randint(0,255,(100,3))

tracks = []

green = (0, 255, 0)
red = (0, 0, 255)

init_flow = True
croppedFirst = True
first = True
stitched = False
if stitched is True:
    stitchedFirst = True
else:
    stitchedFirst = False

ret, frame = cap.read()
frameCount = 0

p0 = None

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
points_video = cv2.VideoWriter('points.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
box_video = cv2.VideoWriter('box.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

#for y in range(0, frame_height - 1, 1):
#    for x in range(0, frame_width - 1, 1):
#        p0 = (y, x)

while(cap.isOpened()):
    prev_frame = frame[:]
    ret, frame = cap.read()
    #frameCount += frameSkipped+1
    #cap.set(1, frameCount)
    if ret:
        im1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        #im1 = cv2.normalize(im1, 0, 255, cv2.NORM_MINMAX)
        #im2 = cv2.normalize(im2, 0, 255, cv2.NORM_MINMAX)

        if filterType is "bilateral":
            im1 = cv2.bilateralFilter(im1, 5, 20, 20)
            im2 = cv2.bilateralFilter(im2, 5, 20, 20)

        # calculate optical flow
        if method is "optical":
            if flowType is "DLK1":
                u, v = optical_flow(im1, im2, 15)
                # Select good points
                vis = frame.copy()
                for x in range (0, frame_width, 10):
                    for y in range(0, frame_height, 10):
                        pt = np.array([y, x])
                        uv = np.array([u[y, x], v[y, x]]) * 4
                        pt2 = ((pt + uv)).astype(np.int)
                        pt = np.flip(pt)
                        pt2 = np.flip(pt2)
                        vis = cv2.arrowedLine(vis, tuple(pt), tuple(pt2), red, 1)

                cv2.imshow("vis", vis)

            elif flowType is "LK":

                # add points to be tracked
                if first is True:
                    p0 = cv2.goodFeaturesToTrack(im2, mask=None, **feature_params)
                    movement_weight = np.zeros_like(im1)
                elif len(p2) < 250:
                    mask = calculate_region_of_interest(im1, p2, 6)
                    cv2.imshow('point mask', mask)
                    p0 = cv2.goodFeaturesToTrack(im2, mask=mask, maxCorners = 250 - len(p2), qualityLevel = 0.001, minDistance = 6, blockSize = 19 )
                    p2 = p2.reshape(-1, 1, 2)
                    p0 = np.concatenate((p0, p2), 0)
                else:
                    p0 = p2.reshape(-1, 1, 2)

                # sparse lucas kanade
                p1, st, err = cv2.calcOpticalFlowPyrLK(im2, im1, p0, None, **lk_params)

                #if a point goes out of frame or near edge, remove it
                i = 0
                for (x0, y0) in p0[:, 0]:
                    if x0 >= frame_width*0.97 or x0 < frame_width*0.03 or y0 >= frame_height*0.97 or y0 < frame_height*0.03:
                        p0 = np.delete(p0, i, 0)
                        p1 = np.delete(p1, i, 0)
                        st = np.delete(st, i, 0)
                        err = np.delete(err, i, 0)
                    else:
                        i = i + 1

                # delete points if they get to close to another point
                tree = cKDTree(p1.reshape(-1,2))
                rows_delete = tree.query_pairs(r=5)
                for p in rows_delete:
                    p0 = np.delete(p0, p, 0)
                    p1 = np.delete(p1, p, 0)
                    st = np.delete(st, p, 0)
                    err = np.delete(err, p, 0)

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                vis = frame.copy()
                flowVectorLength = []
                flowVectorLength_x = []
                flowVectorLength_y = []
                flowAngle = []
                for (x0, y0), (x1, y1), good in zip(p0[:, 0], p1[:, 0], st[:, 0]):
                    if good:
                        cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
                    flowVectorLength.append(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)) # calculate flow vector length (speed)
                    flowVectorLength_x.append(x1 - x0) # calculate the x vector
                    flowVectorLength_y.append(y1 - y0) # calculate the y vector
                    flowAngle.append(math.degrees(math.atan2((y1 - y0), (x1 - x0)))) # calculate flow vector angle (direction)
                    vis = cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
                    cv2.imshow("vis", vis)

                points_video.write(vis)

                # calculate the mean, std, and median of the x, y vectors
                flowVectorLength_average = np.mean(flowVectorLength)
                flowVectorLength_median = np.median(flowVectorLength)
                flowVectorLength_std = np.std(flowVectorLength)
                flowVectorLength_x_average = np.mean(flowVectorLength_x)
                flowVectorLength_x_median = np.median(flowVectorLength_x)
                flowVectorLength_x_std = np.std(flowVectorLength_x)
                flowVectorLength_y_average = np.mean(flowVectorLength_y)
                flowVectorLength_y_median = np.median(flowVectorLength_y)
                flowVectorLength_y_std = np.std(flowVectorLength_y)
                flowAngle_median = np.median(flowAngle)
                h, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
                flowVectorLength_compestated = []
                for i in range(0, len(flowAngle), 1):
                    flowVectorLength_compestated.append(flowVectorLength[i] * math.cos(flowAngle_median - flowAngle[i]))

                #graph all the vectors and angles
                if False:
                    Z = np.vstack((flowVectorLength, flowAngle)).T
                    Z = np.float32(Z)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
                    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    A = Z[label.ravel() == 0]
                    B = Z[label.ravel() == 1]
                    plt.clf()
                    plt.scatter(A[:, 0], A[:, 1])
                    plt.scatter(B[:, 0], B[:, 1], c='r')
                    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
                    plt.axis([0,30,-180,180])
                    plt.xlabel('Distance'), plt.ylabel('Angle')
                    plt.show()

                    fig = plt.figure()
                    ax = fig.add_subplot(111, polar=True)
                    c = ax.scatter(B[:, 1], B[:, 0], c='r')
                    c = ax.scatter(A[:, 1], A[:, 0])

                mask = np.zeros_like(im1)
                i = 0

                # filter out the background
                outliers = []
                inliers = []
                std_tolerance = 1.0
                for (x0, y0), (x1, y1) in zip(p0[:, 0], p1[:, 0]):
                    #if x1 != x0 and y1 != y0:
                        #movement_weight[x1, y1] = movement_weight[x0, y0]
                        #movement_weight[x0, y0] = 0
                    if (flowVectorLength_x[i] > (flowVectorLength_x_median + flowVectorLength_x_std*std_tolerance)) or (flowVectorLength_x[i] < (flowVectorLength_x_median - flowVectorLength_x_std*std_tolerance)) \
                        and ((flowVectorLength_y[i] > (flowVectorLength_y_median + flowVectorLength_y_std*std_tolerance)) or (flowVectorLength_y[i] < (flowVectorLength_y_median - flowVectorLength_y_std*std_tolerance))):
                        #ovement_weight[x1, y1] = movement_weight[x1, y1] + 1
                        outliers.append([x1, y1])
                    else:
                        #ovement_weight[x1, y1] = movement_weight[x1, y1] - 1
                        inliers.append([x1, y1])
                    i = i + 1

                # determine who is likely moving
                if len(outliers) < len(inliers):
                    for (x1, y1) in outliers:
                        cv2.circle(mask, (x1, y1), 13, 255, -1)
                else:
                    for (x1, y1) in inliers:
                        cv2.circle(mask, (x1, y1), 13, 255, -1)

                cv2.imshow("mask", mask)

                kernel = np.ones((11, 11), np.uint8)
                erosion = cv2.erode(mask, kernel, iterations=1)
                cv2.imshow("erosion", erosion)

                # define dilate to fo fill in holes
                dilate = cv2.dilate(mask, kernel, iterations=1)
                cv2.imshow("dilate", dilate)

                contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # loop over the contours
                box = frame.copy()
                i=1
                for c in cnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < contourAreaCutoff:
                        continue

                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(box, (x, y), (x + w, y + h), green, 2)
                    cv2.imshow(("object_" + str(i)), frame[y:y+h, x:x+w]) # display image on screen for debug informatio
                    #obj = frame[y:y + h, x:x + w] # slice object from the full frame
                    if os.path.isdir("images") is False: # check that image dir already exist
                        os.mkdir("images")
                    cv2.imwrite(("images/object_" + str(i) + "_" + str(uid) + ".jpg"), frame[y:y+h, x:x+w]) # write object image to file
                    objClass = objTypeByPath("images/object_" + str(i) + "_" + str(uid) + ".jpg") # pass object image file path to object detector
                    cv2.putText(box, objClass, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2, cv2.LINE_AA) # display detected object class name
                    i=i+1
                    uid=uid+1

                cv2.imshow("box", box)
                box_video.write(box)


                if False:
                    p2 = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
                    #get new points
                    #p2 = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
                    #check if point is already being tracked
                    #p0 = np.concatenate((np.round(p0, 10), np.round(p2, 15)), 0)
                    #p0 = [tuple(row) for row in p0]
                    #p0 = np.asarray(p0)

                p2 = np.concatenate((np.asarray(outliers), np.asarray(inliers)))

            elif flowType is "DLK":
                if first is True:
                    # Create empty matrices to fill later
                    bin_count = 10
                    hist_magnitudes = np.zeros([bin_count, 1])
                    bounds = np.zeros([bin_count, 2])

                # Dense Lucas Kanade
                flow_angles = flow_magnitudes = []
                histogram_temp_mean = np.zeros(bin_count)

                if frame is None:
                    hist_magnitudes = hist_magnitudes / length
                    break

                # Create the old matrix to feed to LK, instead of goodFeaturesToTrack
                #all_pixels = np.where(im1 >= 0)
                #all_pixels = tuple(zip(*all_pixels))
                #all_pixels = np.vstack(all_pixels).reshape(-1, 1, 2).astype("float32")
                all_pixels = []
                for x in range(0, frame_width):
                    for y in range(0, frame_height):
                        all_pixels.append([[x, y]])

                all_pixels = np.asarray(all_pixels).astype("float32")

                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(im2, im1, all_pixels, None, **lk_params)

                # Flow vector
                vis = frame.copy()
                flowVectorLength = []
                flowVectorLength_x = []
                flowVectorLength_y = []
                flowAngle = []
                i = 0
                for (x0, y0), (x1, y1), good in zip(all_pixels[:, 0], p1[:, 0], st[:, 0]):
                    if i % 30 is 0:
                        if good:
                            cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
                        flowVectorLength.append(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
                        flowVectorLength_x.append(x1 - x0)
                        flowVectorLength_y.append(y1 - y0)
                        flowAngle.append(math.atan((y1 - y0) / (x1 - x0)))
                        vis = cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
                    i = i + 1
                cv2.imshow('Dense LK', vis)

            elif flowType is "OF":
                #Dense Optical Flow
                opt_flow = cv2.calcOpticalFlowFarneback(im2, im1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                cv2.imshow('flow', draw_flow(im1, opt_flow))
                flowVectorLength = []
                flowVectorLength_x = []
                flowVectorLength_y = []
                flowAngle = []
                for y in range(0, opt_flow.shape[0] - 1, 1):
                    for x in range(0, opt_flow.shape[1] - 1, 1):
                        flowVectorLength.append(math.sqrt(opt_flow[y, x, 0] ** 2 + opt_flow[y, x, 1] ** 2))
                        flowVectorLength_x.append(opt_flow[y, x, 1])
                        flowVectorLength_y.append(opt_flow[y, x, 0])
                        flowAngle.append(math.atan(opt_flow[y, x, 1] / opt_flow[y, x, 0]))

                flowVectorLength_average = np.mean(flowVectorLength)
                flowVectorLength_std = np.std(flowVectorLength)
                flowVectorLength_x_average = np.mean(flowVectorLength_x)
                flowVectorLength_x_std = np.std(flowVectorLength_x)
                flowVectorLength_y_average = np.mean(flowVectorLength_y)
                flowVectorLength_y_std = np.std(flowVectorLength_y)
                flowAngle_median = np.median(flowAngle)
                fig, axs = plt.subplots(2, 2, figsize=(5, 5))

                off = frame.copy()
                mask = np.zeros_like(im1)
                i = 0
                for y in range(0, opt_flow.shape[0] - 1, 1):
                    for x in range(0, opt_flow.shape[1] - 1, 1):
                        if (flowVectorLength_x[i] > (flowVectorLength_x_average + flowVectorLength_x_std)) or (flowVectorLength_x[i] < (flowVectorLength_x_average - flowVectorLength_x_std)):
                            cv2.circle(mask, (x, y), 10, 255, -1)
                        elif (flowVectorLength_y[i] > (flowVectorLength_y_average + flowVectorLength_y_std)) or (flowVectorLength_y[i] < (flowVectorLength_y_average - flowVectorLength_y_std)):
                            cv2.circle(mask, (x, y), 10, 255, -1)
                        i = i + 1

                # define dilate to fo fill in holes
                mask = cv2.dilate(mask, None, iterations=2)

                contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # loop over the contours
                box = frame.copy()
                for c in cnts:
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < 200:
                        continue

                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(box, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("mask", mask)
                cv2.imshow("box", box)

                kmeans = False
                if kmeans is True:
                    Z = opt_flow.reshape((-1, 2))

                    # convert to np.float32
                    Z = np.float32(Z)

                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    K = 2
                    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    center = np.uint8(center)
                    res = cv2.normalize(label.reshape(im1), 0, 255)
                    result_image = res

                    cv2.imshow("km", result_image)

            elif flowType is "TR":
                flow = cv2.calcOpticalFlowFarneback(im2, im1, None, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                # prevgray = frame_gray

                Z = flow.reshape((-1, 2))

                # convert to np.float32
                Z = np.float32(Z)

                # define criteria, number of clusters(K) and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 2
                ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                # Now convert back into uint8, and make original image
                center = np.uint8(center)
                res = center[label.flatten()]

                res = res[:, 0]
                res = res.reshape((im1.shape))
                new_res = res

                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                vis1 = frame.copy()

                if len(tracks) > 0:
                    img0, img1 = im2, res
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > 15:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                    #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

                mask = np.zeros_like(res)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(res, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])

                # regions = blobdet.detect(new_res)

                # if regions!=[]:
                #   cv2.drawKeypoints(res,regions,vis1, (0,255, 0),4)

                # prev_res = res
                cv2.imshow('lk_track', vis)
                cv2.imshow('blob', vis1)
                cv2.imshow('flow', draw_flow(im1, flow))
                cv2.imshow('res2', res)

                prevgray2c = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

                allf = np.hstack((vis, prevgray2c))
                cv2.imshow('all', allf)

            else:
                if init_flow is True:
                    opt_flow = cv2.calcOpticalFlowFarneback(im2, im1, None, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                    init_flow = False
                else:
                    opt_flow = cv2.calcOpticalFlowFarneback(im2, im1, None, 0.5, 5, 13, 10, 5, 1.1,
                                                            cv2.OPTFLOW_USE_INITIAL_FLOW)
                display_flow(frame, opt_flow)
                flowVectorLength = []
                for y in range(0, opt_flow.shape[0] - 1, 1):
                    for x in range(0, opt_flow.shape[1] - 1, 1):
                        flowVectorLength.append(math.sqrt(opt_flow[y,x,0]**2 + opt_flow[y,x,1]**2))

                flowVectorLength_average = np.mean(flowVectorLength)
                flowVectorLength_stdev = np.std(flowVectorLength)
                off = frame.copy()

                for y in range(0, opt_flow.shape[0] - 1, 1):
                    for x in range(0, opt_flow.shape[1] - 1, 1):
                        if (flowVectorLength[y* (opt_flow.shape[1]-1) + x] > (flowVectorLength_average + flowVectorLength_stdev)):
                            cv2.circle(off, (x, y), 2, (0, 255, 0), -1)
                        elif (flowVectorLength[y* (opt_flow.shape[1]-1) + x] < (flowVectorLength_average - flowVectorLength_stdev)):
                            cv2.circle(off, (x, y), 2, (0, 255, 0), -1)
                cv2.imshow("off",off)
            #mag, ang = cv2.cartToPolar(opt_flow[..., 0], opt_flow[..., 1])
            #mag_avg = sum(mag) / len(mag)
            #mag_std = stdev(mag)
            #for i in mag:
            #    if i < mag_avg + mag_std:
            #        break
        elif method is "feature":
            # detect key feature points
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
            else:
                assert (False, "Invalid Feature Detector")

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
            else:
                assert(False, "Invalid Feature Descriptor")

            if first is True:
                p0 = cv2.goodFeaturesToTrack(im2, mask=None, **feature_params)
                # Create a mask image for drawing purposes
                flowMask = np.zeros_like(im2)
                first = False

            # some magic with prev_frame

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

                imMatches = cv2.drawMatchesKnn(im1, kp1, im2, kp2, [good], None,
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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

            else:
                assert(False, "Invalid Match Type")

            # Find homography
            transformationType = "euclidian"
            if transformationType is "homography":
                h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
                height, width = im2.shape
                im1Reg = cv2.warpPerspective(im1, h, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
                im2Reg = cv2.warpPerspective(im2, h, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
            elif transformationType is "euclidian":
                m = cv2.estimateRigidTransform(points2, points1, fullAffine=False)
                height, width = im2.shape
                im1Reg = cv2.warpAffine(im1, m, (width, height))
                im2Reg = cv2.warpAffine(im2, m, (width, height))
            else:
                assert(False, "Invalid Transformation")
        else:
            assert(False, "Invalid Method")
        """
        else:
            f1, t1, s1 = 0, 260, 10
            f2, t2, s2 = 0, 240, 10
            selected = [(i, 50) for i in range(f1, t1, s1)]
            for j in range(f2, t2, s2):
                selected += [(i, j) for i in range(f1, t1, s1)]
            u, v = optical_flow(im1, im2, 15)
            for i, j in selected:
                pt = np.array([i, j])
                uv = np.array([u[i, j], v[i, j]]) * 4
                pt2 = (pt + uv).astype(np.int)
                cv2.arrowedLine(im1, tuple(pt), tuple(pt2), (255, 0, 0), 1)
            cv2.imshow("optical flow", im1)
        """
        if first is True:
            first = False

    else:
        print('Could not read frame')
        cap.release()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()