import torch
import torch.nn as nn

from ..config import IN_DIM, Z_DIM_AE
from .ae import flat


class VAE(nn.Module):
    def __init__(self, in_dim=IN_DIM, h=256, z_dim=Z_DIM_AE):
        super().__init__()
        self.z_dim = z_dim
        self.body = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU())
        self.fc_mu = nn.Linear(h, z_dim)
        self.fc_logvar = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h), nn.ReLU(),
            nn.Linear(h, in_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.body(flat(x))
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparam(mu, logvar)), mu, logvar
