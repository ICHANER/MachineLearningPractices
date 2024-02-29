import torch

num_inputs = 28 * 28
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
X.sum(0, keepdim=True), X.sum(1,keepdim=True)






# softmax 公式：exp(Xij)/SUM_k(exp(Xik))   X 每行列代表一个样本，每一列代表一个特征
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 按行求和，保持输入输出维度一致，得到列向量
    return X_exp / partition
# 模型定义
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
# 损失函数定义
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)),y])
# 分类精度定义
def accuracy(y_hat, y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())