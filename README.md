# eep596_machine_vision_final

## Motion Detection

SURF and STAR require python-contrib-opencv and are no longer aviliable in opencv > 3.4.2.16

`pip install opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16`

## Object Recognition

###Update 11/24/2019 (Sun):

A fully functional trained 2 layers cnn net is finished, which for now detect 2 classes (car and person) with accuracy close to 100% tested by using 10 images
Implemented a fully functional wrapper function objTypeByPath(path) which take a dir of a single image and return the name of the class predict by trained cnn net
Built training data folder hierarchy with 500 images for each class, and few images for testing (see ./data folder)
Adding single img exception in each class folder by updating .gitignore

###Note:

pls install pillow package by using anaconda

###To do:

Figure out a way to return the probability val to objTypeByPath() too
Add few more layers to cnn net
Find and add some plane class and images to ./data and ./data/test, and re-train the cnn net
Figure out a way to pass pixels of partial image frame by frame from Ryan's moving obj retrieval (partial image is wrapped by Ryan's surrounding box)
(optional): try dilated_conv and depth_conv