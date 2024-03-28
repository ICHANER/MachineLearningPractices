import torch
import numpy as np


class Model:
    # 定义模型
    def logistic(self, X, w, b):
        y = self.sigmoid(torch.matmul(X, w) + b)  # 通过sigmoid函数将线性回归的输出转换为概率
        return y

    # 定义激活函数
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    # 定义损失函数（二元交叉熵损失/对数损失函数）
    def loss(self, y_pred, y_true):
        return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    # 定义优化算法 SGD
    def sgd(self, w, b, lr):
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad


class Train:
    def __init__(self, model, features, labels, lr, epochs):
        self.model = model
        self.features = features
        self.labels = labels
        self.lr = lr
        self.epochs = epochs
        self.w = torch.normal(0, 0.01, size=(features.shape[1], 1), requires_grad=True)  # 权重
        self.b = torch.zeros(1, requires_grad=True)  # 偏置

        # 读取数据集

    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        np.random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]

    # 训练
    def train(self):
        for epoch in range(self.epochs):
            for X, y in self.data_iter(batch_size=10, features=self.features, labels=self.labels):
                y_pred = self.model.logistic(X, self.w, self.b)
                l = Model.loss(model ,y_pred, y)

                # l.backward()
                l.sum().backward()
                Model.sgd(model ,self.w, self.b, self.lr)
            with torch.no_grad():
                y_pred = self.model.logistic(features, self.w, self.b)
                l = Model.loss(model ,y_pred, labels)
                print(f'epoch {epoch + 1}, loss {float(l.mean()):f}')

    def predict(self, X):
        y_pred = self.model.logistic(X, self.w, self.b)  # 预测值
        return y_pred

    # 保存模型
    def save_model(self, path):
        torch.save({
            'w': self.w,
            'b': self.b
        }, path)


class DataProducer:
    # 生成正态线性数据集
    def synthetic_data(self, w, b, num_samples):
        X = torch.normal(0, 1, (num_samples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))

    # 生成二分类数据集
    def synthesize_data_binarization(self, num_examples):
        n_data = torch.ones(num_examples, 2)  # 数据的基本形态 num_examples x 2

        x1 = torch.normal(2 * n_data, 1)  # shape=(50, 2)
        y1 = torch.zeros(num_examples)  # 类型0 shape=(50, 1)

        x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
        y2 = torch.ones(num_examples)  # 类型1 shape=(50, 1)

        # 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
        x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
        y = torch.cat((y1, y2), 0).type(torch.FloatTensor)
        return x, y


if __name__ == '__main__':
    # 准备数据集
    data_producer = DataProducer()
    data = data_producer.synthesize_data_binarization(num_examples=3000)  # 生成1000个样本
    features, labels = data  # 分别获取特征和标签

    # 初始化模型参数
    model = Model()

    # 训练
    batch_size = 100  # 每批数据集大小
    train = Train(model, features, labels, lr=0.003, epochs=3)
    train.train()
