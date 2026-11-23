import torch.nn as nn

from ..config import IN_DIM, Z_DIM_GAN, IMG_SHAPE
from .ae import flat

C_IMG = IMG_SHAPE[0]


class VanillaGAN_G(nn.Module):
    def __init__(self, z_dim=Z_DIM_GAN):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, IN_DIM), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, *IMG_SHAPE)


class VanillaGAN_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(flat(x)).view(-1)


class DCGAN_G(nn.Module):
    def __init__(self, z_dim=Z_DIM_GAN):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, C_IMG, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(-1, self.z_dim, 1, 1))


class DCGAN_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_IMG, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 7, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
