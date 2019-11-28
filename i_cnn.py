#this is the interface from raw .py to trained CNN.ipynb
#Users/TaoYao/Google Drive/UW_Master/MSEE/2019_fall/EE_596/EE596_final/eep596_machine_vision_final

#from CNN.ipynb import test
#from cnn import test
import cv2
import numpy as np

cap = cv2.VideoCapture('video/V3V100003_004.avi')

ret, frame = cap.read()