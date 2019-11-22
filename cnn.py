#import cnn api modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt
import numpy as np

import pdb

#for training
import tqdm
import torch.optim as optim

#init data loader
# transform to do random affine and cast image to PyTorch tensor
trans_ = torchvision.transforms.Compose(
    [
     # torchvision.transforms.RandomAffine(10),
     torchvision.transforms.ToTensor()] #transform from height*width*channel to ch*h*w in order to fit tourch tensor format
)

# Setup the dataset
ds = torchvision.datasets.ImageFolder("train_img/",
                                     transform=trans_)

# Setup the dataloader
loader = torch.utils.data.DataLoader(ds, 
                                     batch_size=16, #batch is how many imgs/samples load per loop
                                     shuffle=True)

# [16, 3, 30, 30] = [batch size, channels, width, height]
for x, y in loader:
    print(x.shape) #the img 
    print(y.shape) #tensor dim; here'll be 16 instead of 16*5
    print(y) #tensor
    break

# vis
for i in range(16):
    plt.imshow(np.transpose(x[i,:], (1,2,0))) # 30 x 30 x 3
    plt.show()

#the cnn class which inherit from torch.nn.Module class
class CNN(nn.Module):
    def __init__(self): #constructor
        super(CNN, self).__init__()
        
        # define the layers
        # kernel size = 3 means (3,3) kernel
        # rgb -> 3 -> in channel
        # number of feature maps = 16
        # number of filters = 3 x 16
        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16) #1st convolve layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #down sampling layer
        # MaxPool2d, AvgPool2d. 
        # The first 2 = 2x2 kernel size, 
        # The second 2 means the stride=2
        
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32) #2nd convolve layer
        
        # FC layer (fully-connected or linear layer)
        self.fc1 = nn.Linear(32 * 6 * 6, 5)
        
    def forward(self, x):
        # define the data flow through the deep learning layers
        #(1st conv->pool layer)
        x = self.pool(F.relu(self.l1(x))) # 16x16 x 14 x 14 
        #(2nd conv->pool layer)
        x = self.pool(F.relu(self.l2(x))) # 16x32x6x6 
        # print(x.shape)
        #flatten layer, set -1 coz last batch might not be full
        x = x.reshape(-1, 32*6*6) # [16 x 1152]# CRUCIAL: 
        # print(x.shape)
        #FC layer
        x = self.fc1(x)
        return x

m = CNN() #init a brand-new untrained cnn
pred = m(x) #prediction
print(pred.shape) #vector will be encoded to 16*5
print(pred)

#now it's time for training
criterion = nn.CrossEntropyLoss()
num_epoches = 50

#our training loop
for epoch_id in range(num_epoches):
    optimizer = optim.SGD(m.parameters(), lr=0.01 * 0.95 ** epoch_id)
    for x, y in tqdm.tqdm(loader):
        optimizer.zero_grad() # clear (reset) the gradient for the optimizer
        pred = m(x)
        loss = criterion(pred, y)
        loss.backward() # calculating the gradient
        optimizer.step() # backpropagation: optimize the model

#now it's time for training
criterion = nn.CrossEntropyLoss()
num_epoches = 50

#our training loop
for epoch_id in range(num_epoches):
    optimizer = optim.SGD(m.parameters(), lr=0.01 * 0.95 ** epoch_id)
    for x, y in tqdm.tqdm(loader):
        optimizer.zero_grad() # clear (reset) the gradient for the optimizer
        pred = m(x)
        loss = criterion(pred, y)
        loss.backward() # calculating the gradient
        optimizer.step() # backpropagation: optimize the model

#after training, test phase test the result include accuracy
# Setup the dataset
test_ds = torchvision.datasets.ImageFolder("test_img/",
                                     transform=trans_)

# Setup the dataloader
testloader = torch.utils.data.DataLoader(test_ds, 
                                     batch_size=16, 
                                     shuffle=True)

all_gt = []
all_pred = []

for x, y in tqdm.tqdm(loader):
    optimizer.zero_grad() # clear (reset) the gradient for the optimizer
    all_gt += list(y.numpy().reshape(-1))
    pred = torch.argmax(m(x), dim=1)
    all_pred += list(pred.numpy().reshape(-1))

print(all_gt)
print(all_pred)
acc = np.sum(np.array(all_gt) == np.array(all_pred)) / len(all_gt)
print("Accuracy is:", acc)

#Dilation and Depth-wise Conv
standard_conv = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=16, dilation=1, groups=1)
#the only difference vs sd_conv is dil_conv has dilation=2
dilated_conv = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=16, dilation=2, groups=1)
#the only difference vs sd_conv is dep_conv has groups=16
depth_conv = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=16, dilation=1, groups=16)
print(sum([p.numel() for p in standard_conv.parameters()]))
print(sum([p.numel() for p in dilated_conv.parameters()]))
print(sum([p.numel() for p in depth_conv.parameters()]))
print(standard_conv.weight.shape)
print(dilated_conv.weight.shape)
print(depth_conv.weight.shape)

