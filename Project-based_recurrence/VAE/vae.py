import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
# import transforms
# import sgmllib
# from html.parser import HTMLParser
# from torchvision.transforms.functional import to_tensor
from torchvision import transforms as ttransforms

# define_model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim*2)  # output avg and std
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # calculate std
        eps = torch.randn_like(std)  # Sample noise from N(0,1)
        z = mu + eps * std  # Reparameterization technique
        return z

    # 前向传播
    def forward(self,x):
        # 编码
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1) #将输出分割为均值和方差
        z = self.reparameterize(mu, logvar)  # 重参数化

        # 解码
        decoded = self.decoder(z)
        return decoded, mu, logvar
