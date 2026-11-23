import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import IN_DIM, Z_DIM_AE


def flat(x):
    return x.reshape(x.size(0), -1)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=IN_DIM, h1=256, z_dim=Z_DIM_AE, n_cls=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, z_dim)
        self.fc3 = nn.Linear(z_dim, n_cls)

    def features(self, x):
        h1 = F.relu(self.fc1(flat(x)))
        return F.relu(self.fc2(h1))

    def forward(self, x):
        return self.fc3(self.features(x))


class MLPDecoder(nn.Module):
    def __init__(self, z_dim=Z_DIM_AE, h=256, out_dim=IN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class FrozenLatentRecon(nn.Module):
    def __init__(self, mlp, z_dim=Z_DIM_AE):
        super().__init__()
        self.mlp = mlp
        self.dec = MLPDecoder(z_dim)

    def train(self, mode=True):
        super().train(mode)
        self.mlp.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            z = self.mlp.features(x)
        return self.dec(z)


class VanillaAE(nn.Module):
    def __init__(self, in_dim=IN_DIM, h=256, z_dim=Z_DIM_AE):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, in_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.enc(flat(x))

    def forward(self, x):
        return self.dec(self.encode(x))


class SparseAE(nn.Module):
    def __init__(self, in_dim=IN_DIM, h=256, z_dim=Z_DIM_AE):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, z_dim), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, in_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.enc(flat(x))

    def forward(self, x):
        return self.dec(self.encode(x))
