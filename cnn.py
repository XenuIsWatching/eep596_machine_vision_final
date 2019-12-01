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
warnings.filterwarnings('ignore')

#init data loader
DATADIR = os.getcwd()+'/data/train_img'
BATCH_SIZE = 16
IMG_SIZE = 100
CENTER_SIZE = IMG_SIZE+IMG_SIZE*0.2
CATAGORIES = ["background","car","person"]
CATEGORY_SIZE = len(CATAGORIES)

# transform to do random affine and cast image to PyTorch tensor
trans_ = torchvision.transforms.Compose(
    [
     # torchvision.transforms.RandomAffine(10),
     torchvision.transforms.Resize((IMG_SIZE)),
     torchvision.transforms.CenterCrop(CENTER_SIZE),
     torchvision.transforms.ToTensor()] #transform from height*width*channel to ch*h*w in order to fit tourch tensor format
)

# Setup the dataset
ds = torchvision.datasets.ImageFolder(root = DATADIR, #tv.dataset will auto find all class folders so just pass data folder
                                     transform=trans_)

# Setup the dataloader
loader = torch.utils.data.DataLoader(ds, 
                                     batch_size=BATCH_SIZE, #batch is how many imgs/samples load per loop
                                     shuffle=True)

type(CATEGORY_SIZE)

#the cnn class which inherit from torch.nn.Module class
layer = 2#4; don't forget to change parameters in final.py when change layer amount!
final_ch = 32 #final out channel, 4 layers has: 128;
class CNN(nn.Module):
    cur_kernel_size = 3
    pool_kernel_val = 2
    cur_img_dim = CENTER_SIZE
    def __init__(self):
        super(CNN, self).__init__()

        self.l1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16) #1st convolve layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #down sampling layer
        self.l2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32) #2nd convolve layer
        #self.l3 = nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64)  # 3rd convolve layer
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
        #x = self.pool(F.relu(self.l3(x)))
        #x = self.pool(F.relu(self.l4(x)))
        #flatten layer
        x = x.view(x.size(0), -1)
        #FC layer
        x = self.fc1(x)
        return x

if os.path.isfile("model.pt") is False:
    m = CNN()
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
    TESTDIR = os.getcwd()+'/data/test_img'
    test_ds = torchvision.datasets.ImageFolder(root = TESTDIR,
                                         transform=trans_)

    # Setup the dataloader
    testloader = torch.utils.data.DataLoader(test_ds,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)

    all_gt = []
    all_pred = []

    for x, y in tqdm.tqdm(testloader):
        optimizer.zero_grad() # clear (reset) the gradient for the optimizer
        all_gt += list(y.numpy().reshape(-1))
        pred = torch.argmax(m(x), dim=1)
        all_pred += list(pred.numpy().reshape(-1))

    print(all_gt)
    print(all_pred)
    acc = np.sum(np.array(all_gt) == np.array(all_pred)) / len(all_gt)
    print("Accuracy is:", acc)

    torch.save(m, "model.pt")
else:
    m = CNN()
    m = torch.load("model.pt")

def image_loader(loader, image_name, printImg = False):
    image = pil.Image.open(image_name)
    image = loader(image).float()
    if printImg == True:
        print(image)
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def objTypeByPath(img_dir):
    idx = np.argmax(m(image_loader(trans_, img_dir)).detach().numpy())
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

print(objTypeByPath("data/test_random/random_test1.jpg"))
print(objTypeByPath("data/test_random/random_test2.jpg"))
print(objTypeByPath("data/test_random/random_test3.jpg"))
print(objTypeByPath("data/test_random/random_test4.jpg"))