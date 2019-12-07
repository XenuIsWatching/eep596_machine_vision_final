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
import pdb
import tqdm
import torch.optim as optim
import os
import PIL as pil
from PIL import Image
import warnings
from scipy.spatial import distance as dist
from collections import OrderedDict
from scipy.spatial import cKDTree
from colorthief import ColorThief

#import cnn as cnn
#from cnn import CNN

warnings.filterwarnings('ignore')

outPyWrite = False # output images to a video
writeToImgFolder = False # output slices images to file

os.environ['KMP_DUPLICATE_LIB_OK']='True' # fix issue with macOS...

uid = 0

# configuration
cap = cv2.VideoCapture('Data_backup/sample_video/V3V100007_017.avi')#Data_backup/sample_video/V3V100003_004.avi
frameSkipped = 1
filterType = "bilateral"
method = "optical"
flowType = "LK"
backgroundFilter = "median"
homographyPoints = True
ransacThreshold = 3.0
pointDistance = 15
contourAreaCutoff = pointDistance * 90
cornerQuality = 0.001
std_tolerance = 1.2
lk_params = dict( winSize =(19, 19),
                  maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners=100,
                       qualityLevel=cornerQuality,
                       minDistance=15,
                       blockSize=19)

# the cnn class which inherit from torch.nn.Module class
### cnn.py start
DATADIR = os.getcwd()+'/data/train_img'
BATCH_SIZE = 16
IMG_SIZE = 100
CENTER_SIZE = IMG_SIZE+IMG_SIZE*0.2#20
CATAGORIES = ["car","motorcycle","person","plane","truck"]
CATEGORY_SIZE = len(CATAGORIES)

# transform to do random affine and cast image to PyTorch tensor
trans_ = torchvision.transforms.Compose(
    [
     # torchvision.transforms.RandomAffine(10),
     torchvision.transforms.Resize((IMG_SIZE)),
     torchvision.transforms.CenterCrop(CENTER_SIZE),
     torchvision.transforms.ToTensor()] #transform from height*width*channel to ch*h*w in order to fit tourch tensor format
)

#the cnn class which inherit from torch.nn.Module class
layer = 3#4; don't forget to change parameters in final.py when change layer amount!
final_ch = 64 #final out channel, 4 layers has: 128;
class CNN(nn.Module):
    cur_kernel_size = 3
    pool_kernel_val = 2
    cur_img_dim = CENTER_SIZE
    def __init__(self):
        super(CNN, self).__init__()

        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16) #1st convolve layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #down sampling layer
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32) #2nd convolve layer
        self.l3 = nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64)  # 3rd convolve layer
        #self.l4 = nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128)  # 4th convolve layer

        #calculate the final dimention (h*w*d) after 2 layers of convolution and downsampling
        global cur_img_dim
        global cur_kernel_size
        global pool_kernel_val
        cur_img_dim = CENTER_SIZE
        cur_kernel_size = 3
        pool_kernel_val = 2
        img_trim = ((cur_kernel_size-1)/2)*2#here assume kernel size is always odd
        for i in range(layer):
            cur_img_dim -= img_trim
            cur_img_dim = cur_img_dim/pool_kernel_val
        cur_img_dim = int(cur_img_dim)

        # FC layer (fully-connected or linear layer)
        self.fc1 = nn.Linear(int(final_ch * cur_img_dim * cur_img_dim), CATEGORY_SIZE) #32 * 28 * 28 for 2 layers

    def forward(self, x):
        #(conv->pool layers)
        x = self.pool(F.relu(self.l1(x)))
        x = self.pool(F.relu(self.l2(x)))
        x = self.pool(F.relu(self.l3(x)))
        #x = self.pool(F.relu(self.l4(x)))
        #flatten layer
        x = x.view(x.size(0), -1)
        #FC layer
        x = self.fc1(x)
        return x

m = CNN()
m = torch.load("model.pt")

def image_loader(loader, image_name, printImg = False):
    image = pil.Image.open(image_name)
    if printImg == True:
        print(image)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def objTypeByPath(img_dir):
    idx = np.argmax(m(image_loader(trans_, img_dir, printImg = True)).detach().numpy())
    return CATAGORIES[idx]

def bgrToRgb(nparrimg):
    for i in range(0,len(nparrimg)):
        temp = nparrimg[i][0]
        nparrimg[i][0] = nparrimg[i][3]
        nparrimg[i][3] = temp
    return nparrimg

#np.array img loader (_trans, nparrimg)
def npArrImg_loader(loader, nparrimg, printImg = False):
    nparrimg = bgrToRgb(nparrimg)
    image = pil.Image.fromarray(nparrimg.astype('uint8'), 'RGB')
    if printImg == True:
        print(image)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def objTypeByNpImg(nparrimg):
    idx = np.argmax(m(npArrImg_loader(trans_, nparrimg)).detach().numpy())
    return CATAGORIES[idx]

### cnn.py end

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
if outPyWrite is True:
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
            if flowType is "LK":
                # add points to be tracked
                if first is True:
                    p0 = cv2.goodFeaturesToTrack(im2, mask=None, **feature_params)
                    movement_weight = np.zeros_like(im1)
                # if points get low, add more..
                elif len(p2) < 250:
                    mask = calculate_region_of_interest(im1, p2, pointDistance) # mask out existing points
                    cv2.imshow('point mask', mask)
                    p0 = cv2.goodFeaturesToTrack(im2, mask=mask, maxCorners = 300 - len(p2), qualityLevel = cornerQuality, minDistance = pointDistance, blockSize = 19 )
                    p2 = p2.reshape(-1, 1, 2)
                    # combine old points with the new points
                    if p0 is not None:
                        p0 = np.concatenate((p0, p2), 0)
                    else:
                        p0 = p2
                else:
                    p0 = p2.reshape(-1, 1, 2)

                # if not many points are being found, lower the quality
                if p0.shape[0] < 200:
                    cornerQuality = cornerQuality - cornerQuality/2
                # if too many points are being found, increase the quality
                elif p0.shape[0] > 275 and cornerQuality < 0.5:
                    cornerQuality = cornerQuality + cornerQuality

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
                rows_delete = tree.query_pairs(r=pointDistance*0.5)
                for p in rows_delete:
                    p0 = np.delete(p0, p, 0)
                    p1 = np.delete(p1, p, 0)
                    st = np.delete(st, p, 0)
                    err = np.delete(err, p, 0)

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # calculate the Homography matrix for frame t and t-1
                h, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransacThreshold)

                # add 1 column for z with x y vector matrix
                o = np.ones(p1.shape[0]) # create matrix full of 1
                o = o.reshape(-1,1) # reshape to vertical
                p0z = np.dstack((p0, o)) # add column to xy vector
                pT = p0.copy() # copy original points (really not necessary b/c overwritten later..)
                iter = 0
                for p in p0z:
                    v = np.matmul(h, p.T) # multiply homography matrix with xyz vertical matrix
                    v = v.T # Transpose for easier delete
                    v[0, 0] = v[0, 0] / v[0, 2] # x / z (planar)
                    v[0, 1] = v[0, 1] / v[0, 2] # y / z
                    v = np.delete(v, 2, 1) # remove z axis
                    v = v.reshape((1,1,2))
                    pT[iter,:,:] = v
                    iter = iter + 1

                vis = frame.copy()
                flowVectorLength = []
                flowVectorLength_x = []
                flowVectorLength_y = []
                flowAngle = []
                flowVectorLength_H = []
                flowVectorLength_xH = []
                flowVectorLength_yH = []
                flowAngle_H = []
                for (x0, y0), (x1, y1), (xT, yT), good in zip(p0[:, 0], p1[:, 0], pT[:, 0], st[:, 0]):
                    if good:
                        cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
                        flowVectorLength_H.append(math.sqrt((x1 - xT) ** 2 + (y1 - yT) ** 2)) # calculate flow vector length (speed)
                        flowVectorLength_xH.append(x1 - xT) # calculate the x vector
                        flowVectorLength_yH.append(y1 - yT) # calculate the y vector
                        flowAngle_H.append(math.degrees(math.atan2((y1 - yT), (x1 - xT)))) # calculate flow vector angle (direction)
                        flowVectorLength.append(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)) # calculate flow vector length (speed)
                        flowVectorLength_x.append(x1 - x0) # calculate the x vector
                        flowVectorLength_y.append(y1 - y0) # calculate the y vector
                        flowAngle.append(math.degrees(math.atan2((y1 - y0), (x1 - x0)))) # calculate flow vector angle (direction)
                    vis = cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
                    #vis = cv2.circle(vis, (x0, y0), 2, (red, green)[good], -1)
                    vis = cv2.circle(vis, (xT, yT), 2, (255, 0, 0), -1)

                # get distance, and store in polar format
                x = p1[:, 0, 0] - p0[:, 0, 0]
                y = p1[:, 0, 1] - p0[:, 0, 1]
                mag, ang = cv2.cartToPolar(x, y, angleInDegrees=False)
                im2Reg = cv2.warpPerspective(im2, h, (frame_width, frame_height), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

                x = p1[:, 0, 0] - pT[:, 0, 0]
                y = p1[:, 0, 1] - pT[:, 0, 1]
                magH, angH = cv2.cartToPolar(x, y, angleInDegrees=False)


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
                ##
                flowVectorLength_H_average = np.mean(flowVectorLength_H)
                flowVectorLength_H_median = np.median(flowVectorLength_H)
                flowVectorLength_H_std = np.std(flowVectorLength_H)
                flowVectorLength_xH_average = np.mean(flowVectorLength_xH)
                flowVectorLength_xH_median = np.median(flowVectorLength_xH)
                flowVectorLength_xH_std = np.std(flowVectorLength_xH)
                flowVectorLength_yH_average = np.mean(flowVectorLength_yH)
                flowVectorLength_yH_median = np.median(flowVectorLength_yH)
                flowVectorLength_yH_std = np.std(flowVectorLength_yH)
                flowAngle_H_median = np.median(flowAngle_H)


                #graph all the vectors and angles
                if True:
                    Z = np.vstack((flowVectorLength, flowAngle)).T
                    Z = np.float32(Z)
                    rcolors = ang
                    #fig = plt.figure()
                    plt.clf()
                    bx = plt.subplot(221, polar=True)
                    bx.scatter(ang, mag, c=rcolors)
                    bx.set_title("XY vector before Homography (polar)")

                    cx = plt.subplot(222)
                    cx.set_title("XY Histogram vector before Homography (cart)")
                    cx.set_xlabel("X vector")
                    cx.set_ylabel("Y vector")
                    cx.hist2d(flowVectorLength_x, flowVectorLength_y, bins=(25, 25), cmap=plt.cm.jet)
                    cx.scatter(flowVectorLength_x_average, flowVectorLength_y_average)
                    cx.annotate("mean", (flowVectorLength_x_average, flowVectorLength_y_average))
                    cx.scatter(flowVectorLength_x_median, flowVectorLength_y_median)
                    cx.annotate("median", (flowVectorLength_x_median, flowVectorLength_y_median))
                    if backgroundFilter is "median":
                        stdRect = plt.Rectangle((flowVectorLength_x_median - flowVectorLength_x_std,
                                                 flowVectorLength_y_median - flowVectorLength_y_std),
                                                flowVectorLength_x_std * 2, flowVectorLength_y_std * 2, color='r',
                                                fill=False)
                    elif backgroundFilter is "mean":
                        stdRect = plt.Rectangle((flowVectorLength_x_average - flowVectorLength_x_std,
                                                 flowVectorLength_y_average - flowVectorLength_y_std),
                                                flowVectorLength_x_std * 2, flowVectorLength_y_std * 2, color='r',
                                                fill=False)
                    cx.add_artist(stdRect)

                    ax = plt.subplot(223, polar=True)
                    ax.scatter(angH, magH, c=rcolors)
                    ax.set_title("XY vector after Homography (polar)")

                    vx = plt.subplot(224)
                    vx.set_title("XY Histogram vector after Homography (cart)")
                    vx.set_xlabel("X vector")
                    vx.set_ylabel("Y vector")
                    vx.hist2d(flowVectorLength_xH, flowVectorLength_yH, bins=(25, 25), cmap=plt.cm.jet)
                    vx.scatter(flowVectorLength_xH_average, flowVectorLength_yH_average)
                    vx.annotate("mean",(flowVectorLength_xH_average, flowVectorLength_yH_average))
                    vx.scatter(flowVectorLength_xH_median, flowVectorLength_yH_median)
                    vx.annotate("median", (flowVectorLength_xH_median, flowVectorLength_yH_median))
                    if backgroundFilter is "median":
                        stdRect = plt.Rectangle((flowVectorLength_xH_median - flowVectorLength_xH_std,
                                                 flowVectorLength_yH_median - flowVectorLength_yH_std),
                                                flowVectorLength_xH_std * 2, flowVectorLength_yH_std * 2, color='r',
                                                fill=False)
                    elif backgroundFilter is "mean":
                        stdRect = plt.Rectangle((flowVectorLength_xH_average - flowVectorLength_xH_std,
                                                 flowVectorLength_yH_average - flowVectorLength_yH_std),
                                                flowVectorLength_xH_std * 2, flowVectorLength_yH_std * 2, color='r',
                                                fill=False)
                    vx.add_artist(stdRect)

                mask = np.zeros_like(im1)
                i = 0

                # filter out the background, by using a median (or mean) threshold
                outliers = []
                inliers = []
                for (x0, y0), (x1, y1), good in zip(p0[:, 0], p1[:, 0], st[:, 0]):
                    if good:
                        if backgroundFilter is "median":
                            if homographyPoints is True:
                                if (flowVectorLength_xH[i] > (
                                        flowVectorLength_xH_median + flowVectorLength_xH_std * std_tolerance)) or (
                                        flowVectorLength_xH[i] < (
                                        flowVectorLength_xH_median - flowVectorLength_xH_std * std_tolerance)) \
                                        and ((flowVectorLength_yH[i] > (
                                        flowVectorLength_yH_median + flowVectorLength_yH_std * std_tolerance)) or (
                                                     flowVectorLength_yH[i] < (
                                                     flowVectorLength_yH_median - flowVectorLength_yH_std * std_tolerance))):
                                    outliers.append([x1, y1])
                                else:
                                    inliers.append([x1, y1])
                            else:
                                if (flowVectorLength_x[i] > (
                                        flowVectorLength_x_median + flowVectorLength_x_std * std_tolerance)) or (
                                        flowVectorLength_x[i] < (
                                        flowVectorLength_x_median - flowVectorLength_x_std * std_tolerance)) \
                                        and ((flowVectorLength_y[i] > (
                                        flowVectorLength_y_median + flowVectorLength_y_std * std_tolerance)) or (
                                                     flowVectorLength_y[i] < (
                                                     flowVectorLength_y_median - flowVectorLength_y_std * std_tolerance))):
                                    outliers.append([x1, y1])
                                else:
                                    inliers.append([x1, y1])
                        elif backgroundFilter is "mean":
                            if homographyPoints is True:
                                if (flowVectorLength_xH[i] > (flowVectorLength_xH_average + flowVectorLength_xH_std*std_tolerance)) or (flowVectorLength_xH[i] < (flowVectorLength_xH_average - flowVectorLength_xH_std*std_tolerance)) \
                                    and ((flowVectorLength_yH[i] > (flowVectorLength_yH_average + flowVectorLength_yH_std*std_tolerance)) or (flowVectorLength_yH[i] < (flowVectorLength_yH_average - flowVectorLength_yH_std*std_tolerance))):
                                    outliers.append([x1, y1])
                                else:
                                    inliers.append([x1, y1])
                            else:
                                if (flowVectorLength_x[i] > (flowVectorLength_x_average + flowVectorLength_x_std*std_tolerance)) or (flowVectorLength_x[i] < (flowVectorLength_x_average - flowVectorLength_x_std*std_tolerance)) \
                                    and ((flowVectorLength_y[i] > (flowVectorLength_y_average + flowVectorLength_y_std*std_tolerance)) or (flowVectorLength_y[i] < (flowVectorLength_y_average - flowVectorLength_y_std*std_tolerance))):
                                    outliers.append([x1, y1])
                                else:
                                    inliers.append([x1, y1])
                        i = i + 1

                # determine who is likely moving
                if len(outliers) < len(inliers):
                    for (x1, y1) in outliers:
                        cv2.circle(mask, (x1, y1), int(pointDistance*0.8), 255, -1)
                else:
                    for (x1, y1) in inliers:
                        cv2.circle(mask, (x1, y1), int(pointDistance*0.8), 255, -1)

                cv2.imshow("mask", mask)

                kernel = np.ones((10, 10), np.uint8)

                # define dilate to fo fill in holes
                dilate = cv2.dilate(mask, kernel, iterations=1)
                cv2.imshow("dilate", dilate)

                cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # loop over the contours
                box = frame.copy()
                i=1
                isBackground = False
                cntArea = []
                for c in cnts:
                    # if the contour is too small, ignore it
                    cntArea.append(cv2.contourArea(c))
                    if cv2.contourArea(c) < contourAreaCutoff:
                        continue

                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w, h) = cv2.boundingRect(c)
                    #cv2.imshow(("object_" + str(i)), frame[y:y+h, x:x+w]) # display image on screen for debug information
                    #np array image
                    npArrImg = frame[y:y + h, x:x + w] # slice object from the full frame
                    if writeToImgFolder is True:
                        if os.path.isdir("images") is False: # check that image dir already exist
                           os.mkdir("images")
                        cv2.imwrite(("images/object_" + str(i) + "_" + str(uid) + ".jpg"), frame[y:y+h, x:x+w]) # write object image to file
                    objClass = objTypeByNpImg(npArrImg)
					
					# !!! START : Let's try to get the dominant color of a detected object in each sliced image/file!
                    """
                    image = cv2.imread("images/object_" + str(i) + "_" + str(uid) + ".jpg")
                    original = image.copy()
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                    canny = cv2.Canny(blurred, 120, 255, 1)
                    kernel = np.ones((5, 5), np.uint8)
                    dilate = cv2.dilate(canny, kernel, iterations=1)

                    # Find contours
                    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                    # Iterate thorugh contours and filter for ROI
                    if os.path.isdir("contours") is False: # check that image dir already exist
                        os.mkdir("contours")
                    image_number = 0
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        ROI = original[y:y + h, x:x + w]
                        cv2.imwrite("contours/ROI_" + str(i) + "_" + str(uid) + ".jpg", ROI)
                        if image_number == 0:
                            break
                        #image_number += 1

                    color_thief = ColorThief("contours/ROI_" + str(i) + "_" + str(uid) + ".jpg")
                    """
                    im = Image.open("images/object_" + str(i) + "_" + str(uid) + ".jpg")
                    width, height = im.size  # Get dimensions
                    #print("Width : " + str(width))
                    #print("Height : " + str(height))
                    new_height = height / 2
                    new_width = width / 2

                    left = (width - new_width) / 2
                    top = (height - new_height) / 2
                    right = (width + new_width) / 2
                    bottom = (height + new_height) / 2

                    # Crop the center of the image
                    if os.path.isdir("cropped") is False:  # check that image dir already exist
                        os.mkdir("cropped")
                    im = im.crop((left, top, right, bottom))
                    im.save('cropped/crop_' + str(i) + '_' + str(uid) + '.jpg')
                    #cv2.imwrite("cropped/crop_" + str(i) + "_" + str(uid) + ".jpg", im)

                    color_thief = ColorThief("cropped/crop_" + str(i) + "_" + str(uid) + ".jpg")
                    dominant_color = color_thief.get_color(quality=1)  # (R, G, B) format
                    # !!! END : Finding the dominant color of an image

                    #if objClass is "background":
                        #isBackground = True
                    #if isBackground is False:
                    cv2.rectangle(box, (x, y), (x + w, y + h), green, 2)
                    cv2.putText(box, objClass, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, dominant_color, 2, cv2.LINE_AA) # display detected object class name
                    i=i+1
                    uid=uid+1

                cntAreaMean = np.mean(cntArea)

                # write the number of points in the corner
                cv2.putText(vis, "total points: " + str(p1.shape[0]), (5, 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, green, 1,
                            cv2.LINE_AA)
                cv2.putText(vis, "background: " + str(len(inliers)), (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.25, green, 1,
                            cv2.LINE_AA)
                cv2.putText(vis, "foreground: " + str(len(outliers)), (5, 26), cv2.FONT_HERSHEY_DUPLEX, 0.25, green, 1,
                            cv2.LINE_AA)
                cv2.putText(vis, "corner quality: " + str(cornerQuality), (5, 34), cv2.FONT_HERSHEY_DUPLEX, 0.25, green, 1,
                            cv2.LINE_AA)
                cv2.putText(vis, "point distance: " + str(pointDistance), (5, 42), cv2.FONT_HERSHEY_DUPLEX, 0.25, green,
                            1,
                            cv2.LINE_AA)
                cv2.imshow("vis", vis)

                cv2.imshow("box", box)
                if outPyWrite is True:
                    points_video.write(vis)
                    box_video.write(box)

                plt.draw()
                plt.show(block=False)

                if False:
                    p2 = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
                    #get new points
                    #p2 = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
                    #check if point is already being tracked
                    #p0 = np.concatenate((np.round(p0, 10), np.round(p2, 15)), 0)
                    #p0 = [tuple(row) for row in p0]
                    #p0 = np.asarray(p0)

                if len(outliers) is 0:
                    p2 = np.asarray(inliers)
                else:
                    p2 = np.concatenate((np.asarray(outliers), np.asarray(inliers)))

                # keys presses to increase/decrease point distances while running
                c = cv2.waitKey(1)
                if 'w' == chr(c & 255):
                    pointDistance = pointDistance + 1
                elif 's' == chr(c & 255):
                    if pointDistance > 1:
                        pointDistance = pointDistance - 1

            # experiment with dense optical flow
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

        # experirement using a feature based dectection
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
                h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, ransacThreshold)
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