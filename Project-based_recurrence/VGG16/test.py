import torch
from torch import ones
from VGG16Net import *

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn,optim,device
# from torch.utils.tensorboard import SummaryWriter

dev = 'mps'

device=device(dev)

train_transform = transforms.Compose([
	        transforms.ToTensor(),
	        # 归一化
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    	])

valid_transform=transforms.Compose([
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       	    ])
train_dataset =torchvision.datasets.CIFAR10(root='./data',train=True,transform=train_transform,download=False)
valid_dataset =torchvision.datasets.CIFAR10(root='./data',train=False,transform=valid_transform,download=False)

batch_size=1
train_loader =DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
valid_loader =DataLoader(valid_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
print('train_dataset',len(train_dataset))  #50000
print('valid_dataset',len(valid_dataset))  #10000


batch_size=1

model = VGG16()

model = model.to(torch.device(dev))

input=ones((batch_size,3,32,32))
input=input.to(device=dev)  #调用GPU
output=model(input)
