import random
import torch
# from d2l import torch as d2l

# 数据集生成函数
def synthetic_data(w, b, num_examples):
    X = torch.normal(0 ,1, (num_examples, len(w)))  # 生成一个形状为(num_examples, len(w))的张量X，其中每个元素都从N(0,1)随机采样而来,这个张量包含了输入特征
    y = torch.matmul(X,w) + b    # 模型预测结果
    y += torch.normal(0, 0.01, y.shape)  # 为了增加一些噪声，给预测值y加上N(0,0.01)的随机噪声。添加噪声模拟真实数据环境 增加数据的多样性 防止过拟合 提高模型泛化能力。
    return X, y.reshape((-1, 1)) # 将特征张量与标签张量返回。reshape将(n,)张量调整为(n,1)，保持了标签的一致性，更好与特征张量进行匹配

# 数据集读取函数：按照指定的批量大小从特征和标签中获取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)    # 获取特征张量长度，即样本数
    indices = list(range(num_examples))   # 创建一个包含从 0 到 num_examples - 1 的索引列表。
    random.shuffle(indices)         # 将索引列表随机打乱，以确保每个批次的样本是随机的
    for i in range(0, num_examples, batch_size):    # 循环遍历索引列表，每次取出一个批次的索引。从 0 到 num_examples - 1 数字序列，步长 batch_size
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # ⬆根据当前批次的起始索引和批量大小，取出对应的批次索引。需要注意的是，当剩余的样本不足一个批次时，取出的索引可能不足一个批次，这里使用 min() 函数来确保不超出索引范围
        # 从索引 i 开始，取出一个长度为 min(batch_size, num_examples - i) 的切片
        yield features[batch_indices], labels[batch_indices]
        # 使用yield连续返回

# 模型定义
def linreg(X,w,b):
    return torch.matmul(X,w) + b

# 损失函数定义
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():  # 临时禁用梯度计算
        for param in params:
            param -= lr * param.grad / batch_size
            # 除以批量大小（batch_size）可以将每个样本的梯度值进行平均，使得参数更新的步长更为合理。这样做的目的是使得参数更新的规模不会受到批量大小的影响，从而保持了参数更新的一致性，同时也有助于控制学习算法的稳定性。
            param.grad.zero_()    # 调用 .zero_() 方法将梯度清零


### 训练 ###

# 数据集生成参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,10000)

#每一批数据个数
batch_size = 10

# 参数初始化
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练参数设定
lr = 0.03             # 学习率
num_epochs = 3        # 训练次数
net = linreg          # 模型
loss = squared_loss   # 损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()                 # 计算损失函数 l 对模型参数的梯度，并将这些梯度累加到参数的 .grad 属性中。
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')




