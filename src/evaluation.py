import time

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

from .config import IMG_SHAPE
from .models.ae import flat


@torch.no_grad()
def evaluate_recon(recon_fn, loader, device):
    mses = []
    psnrs = []
    ssims = []
    for images, _ in loader:
        images = images.to(device)
        reconstructed = recon_fn(images).clamp(0, 1)
        mse = ((reconstructed - flat(images)) ** 2).mean(1)
        mses.append(mse.cpu())
        psnrs.append(
            (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu()
        )
        original_np = images.cpu().numpy()
        reconstructed_np = reconstructed.reshape(images.shape).cpu().numpy()
        for index in range(original_np.shape[0]):
            if IMG_SHAPE[0] == 1:
                score = sk_ssim(
                    original_np[index, 0],
                    reconstructed_np[index, 0],
                    data_range=1.0,
                )
            else:
                score = sk_ssim(
                    original_np[index],
                    reconstructed_np[index],
                    data_range=1.0,
                    channel_axis=0,
                )
            ssims.append(score)

    return {
        "mse": torch.cat(mses).mean().item(),
        "psnr": torch.cat(psnrs).mean().item(),
        "ssim": float(np.mean(ssims)),
    }


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_mlp(model, loaders, device, epochs=30, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}
    best_val = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = count = 0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)

        val_acc = accuracy(model, loaders["val"], device)
        history["train_loss"].append(total_loss / count)
        history["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"Epoch {epoch:02d} | loss {total_loss/count:.4f} "
            f"| val acc {val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    test_acc = accuracy(model, loaders["test"], device)
    print(f"\nTEST ACCURACY: {test_acc:.4f}")
    return history, test_acc


def orth_error(weight):
    gram = weight @ weight.T
    identity = torch.eye(gram.size(0), device=weight.device)
    return ((gram - identity).norm() / identity.norm()).item()


def measure_latency(function, warmup=10, runs=50, device=None):
    with torch.no_grad():
        for _ in range(warmup):
            function()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            function()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000
