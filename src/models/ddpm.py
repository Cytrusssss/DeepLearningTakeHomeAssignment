import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import IMG_SHAPE, T_DIFF

C_IMG = IMG_SHAPE[0]


def make_diffusion_schedule(device, timesteps=T_DIFF):
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_ab": alpha_bar.sqrt(),
        "sqrt_1m_ab": (1.0 - alpha_bar).sqrt(),
        "sqrt_recip_alpha": (1.0 / alphas).sqrt(),
    }


def q_sample(x0, t, noise, schedule):
    sqrt_ab = schedule["sqrt_ab"]
    sqrt_1m_ab = schedule["sqrt_1m_ab"]
    return sqrt_ab[t].view(-1, 1, 1, 1) * x0 + sqrt_1m_ab[t].view(-1, 1, 1, 1) * noise


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(te)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, c_in=C_IMG, base=64, tdim=256):
        super().__init__()
        self.temb = nn.Sequential(
            nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )
        self.inc = nn.Conv2d(c_in, base, 3, 1, 1)
        self.enc1 = ResBlock(base, base, tdim)
        self.down1 = nn.Conv2d(base, base, 3, 2, 1)
        self.enc2 = ResBlock(base, base * 2, tdim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, 2, 1)
        self.mid1 = ResBlock(base * 2, base * 4, tdim)
        self.mid2 = ResBlock(base * 4, base * 2, tdim)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)
        self.dec2 = ResBlock(base * 4, base * 2, tdim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.dec1 = ResBlock(base * 2, base, tdim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, c_in, 3, 1, 1)
        )

    def forward(self, x, t):
        te = self.temb(timestep_embedding(t, 128))
        h0 = self.inc(x)
        h1 = self.enc1(h0, te)
        h2 = self.enc2(self.down1(h1), te)
        m = self.mid2(self.mid1(self.down2(h2), te), te)
        u2 = self.dec2(torch.cat([self.up2(m), h2], 1), te)
        u1 = self.dec1(torch.cat([self.up1(u2), h1], 1), te)
        return self.out(u1)


@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    for ema_param, model_param in zip(ema.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(model_param.detach(), alpha=1 - decay)
