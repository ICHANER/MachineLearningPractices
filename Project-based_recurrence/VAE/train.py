from vae import *

#定义训练函数
def train_vae(model, train_loader,num_epochs, learning_rate):
    criterion = nn.BCELoss() # 二分类交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

    model.train()  # 设置模型为训练模式

    for epoch in range(num_epochs):
        total_loss = 0.0

        for data in train_loader:
            images, _ = data
            images = images.view(images.size(0), -1)  # 展平输入图像

            optimizer.zero_grad()

            # 前向传播
            outputs, mu, logvar = model(images)

            # 计算重构损失和KL散度
            reconstruction_loss = criterion(outputs, images)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # 计算总损失
            loss = reconstruction_loss + kl_divergence

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 输出当前训练轮次的损失
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))

    print('Training finished.')


# main
if __name__ == '__main__':
    # 设置超参数
    input_dim = 784  # 输入维度（MNIST图像的大小为28x28，展平后为784）
    hidden_dim = 256  # 隐层维度
    latent_dim = 64  # 潜在空间维度
    num_epochs = 10  # 训练轮次
    learning_rate = 0.001  # 学习率

    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=ttransforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

    # 创建VAE模型
    model = VAE(input_dim, hidden_dim, latent_dim)
    device = torch.device('cpu')
    model = model.to(device)
    # 训练VAE模型
    train_vae(model, train_loader, num_epochs, learning_rate)