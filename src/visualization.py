import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from .config import FIG_DIR


def to_img(tensor):
    return tensor.reshape(-1, *tensor.shape[1:]).cpu()


def show_row(axes, images, ylabel):
    for axis, image in zip(axes, images):
        axis.imshow(image.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    axes[0].set_ylabel(ylabel, fontsize=9)


def plot_sample_grids(samplers, names, filename="fig2_random_samples.png"):
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.4))
    for index, (axis, name) in enumerate(zip(np.atleast_1d(axes), names)):
        grid = make_grid(
            samplers[name](16).cpu(),
            nrow=4, padding=1, pad_value=1.0,
        )
        axis.imshow(grid.permute(1, 2, 0).numpy(), vmin=0, vmax=1)
        axis.set_title(f"Panel {chr(65 + index)}: {name}", fontsize=10)
        axis.axis("off")
    fig.suptitle("Figur 2. Sampling acak 4x4 (seed tetap)", fontsize=11)
    plt.tight_layout()
    output = FIG_DIR / filename
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] disimpan ke {output}")


def curves(histories, title, series, ylabel="loss", filename=None):
    fig, axis = plt.subplots(figsize=(7, 4))
    for name, history_key in series:
        axis.plot(histories[name][history_key], label=name)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.legend()
    plt.tight_layout()
    if filename:
        fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
