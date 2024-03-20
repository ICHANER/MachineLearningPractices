# from torchinfo import summary
# summary(vgg, input_size=(20,3,224,224))
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from VGG16Net import *
# 生成随机输入数据
input_data = torch.randn(64, 3, 32, 32)

# 将输入数据传入模型
model = VGG16()
output = model(input_data)