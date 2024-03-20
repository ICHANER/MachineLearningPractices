import time
import torch
from torch import ones
from VGG16Net import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn,optim,device
from VGG16Net import *

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn,optim,device
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda")  # 使用第一块GPU

model = VGG16()
model = nn.DataParallel(model)  # 将模型包装在DataParallel中
model.to(device)  # 将模型移动到所选设备

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

train_dataset =torchvision.datasets.CIFAR10(root=r'./data',train=True,transform=train_transform,download=True)
valid_dataset =torchvision.datasets.CIFAR10(root=r'./data',train=False,transform=valid_transform,download=True)

batch_size=1


train_loader =DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
valid_loader =DataLoader(valid_dataset,batch_size=batch_size, shuffle=True,num_workers=0)
print('train_dataset',len(train_dataset))  #50000
print('valid_dataset',len(valid_dataset))  #10000



# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
lr = 0.01

# 每n次epoch更新一次学习率
step_size = 2
# momentum(float)-动量因子
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
schedule = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

train_step = 0
vali_step = 0
writer = SummaryWriter('./logs')
epoch = 5

for i in range(1, epoch + 1):
    starttime = time.time()
    train_acc = 0
    # -----------------------------训练过程------------------------------
    for data in train_loader:
        img, tar = data
        img = img.to(device)
        tar = tar.to(device)

        outputs = model(img)
        train_loss = loss_fn(outputs, tar)

        # print(outputs.argmax(1),tar)
        train_acc += sum(outputs.argmax(1) == tar) / batch_size
        # 优化器优化模型
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 5000 == 0:
            print("train_step", train_step)
    vali_loss = 0
    vali_acc = 0
    # -----------------------------验证过程------------------------------
    with torch.no_grad():
        for vali_data in valid_loader:
            img, tar = vali_data
            img = img.to(device)
            tar = tar.to(device)
            outputs = model(img)
            vali_loss += loss_fn(outputs, tar)
            vali_acc += sum(outputs.argmax(1) == tar) / batch_size

            vali_step += 1
            if vali_step % 2000 == 0:
                print("vali_step", vali_step)

    endtime = time.time()
    spendtime = endtime - starttime
    ave_train_acc = train_acc / (len(train_loader))
    ave_vali_loss = vali_loss / (len(valid_loader))
    ave_vali_acc = vali_acc / (len(valid_loader))
    # 训练次数：每一个epoch就是所有train的图跑一遍：1968*3/batch_size,每次batch_size张图
    print("Epoch {}/{} : train_step={}, vali_step={}, spendtime={}s".format(i, epoch, train_step, vali_step, spendtime))
    print("ave_train_acc={}, ave_vali_acc={}".format(ave_train_acc, ave_vali_acc))
    print("train_loss={}, ave_vali_loss={} \n".format(train_loss, ave_vali_loss))

    # tensorboard --logdir=logs
    with SummaryWriter('./logs/ave_train_acc') as writer:
        writer.add_scalar('Acc', ave_train_acc, i)
    with SummaryWriter('./logs/ave_vali_acc') as writer:
        writer.add_scalar('Acc', ave_vali_acc, i)
    with SummaryWriter('./logs/train_loss') as writer:
        writer.add_scalar('Loss', train_loss, i)
    with SummaryWriter('./logs/ave_vali_loss') as writer:
        writer.add_scalar('Loss', ave_vali_loss, i)

torch.save(model.state_dict(), 'model.pth')
writer.close()
