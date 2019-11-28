# eep596_machine_vision_final

## Motion Detection

SURF and STAR require python-contrib-opencv and are no longer aviliable in opencv > 3.4.2.16

`pip install opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16`

## Object Recognition

##Update 11/24/2019 (Sun):

A fully functional trained 2 layers cnn net is finished, which for now detect 2 classes (car and person) with accuracy close to 100% tested by using 10 images
Implemented a fully functional wrapper function objTypeByPath(path) which take a dir of a single image and return the name of the class predict by trained cnn net
Built training data folder hierarchy with 500 images for each class, and few images for testing (see ./data folder)
Adding single img exception in each class folder by updating .gitignore

##Note:

pls install pillow package by using anaconda

##To do:

Figure out a way to return the probability val to objTypeByPath() too
Add few more layers to cnn net
Find and add some plane class and images to ./data and ./data/test, and re-train the cnn net
Figure out a way to pass pixels of partial image frame by frame from Ryan's moving obj retrieval (partial image is wrapped by Ryan's surrounding box)
(optional): try dilated_conv and depth_conv

##notes:
To do:
11/22/2019 (Fri):
1. find and organize data folder, load and visualize out all imgs
2. feed imgs to 2 layer cnn and test result
3. Learnt unknown concepts from Questions

11/23-11/24 (Sat,Sun):
1. Figure out a way to return obj type and pass it to moving obj
2. add more layers (deeper network), higher resolution img (300*300)
3. Use Dilation and depth conv

Questions:
  1. What is tensor?
  2. in class CNN, how to determin what kind of conv kernels each layer is using?
  3. epoch, criterion, optim.SGD, cross-entropy loss?
  4. Dilation and depth conv?