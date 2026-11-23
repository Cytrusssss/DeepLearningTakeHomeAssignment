from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE, DATASET, DATA_ROOT, NUM_WORKERS, SEED,
    SPLIT_FILE, TRAIN_RATIO, VAL_RATIO,
)
from .utils import load_json, save_json


DATASETS = {
    "FashionMNIST": datasets.FashionMNIST,
    "MNIST": datasets.MNIST,
    "KMNIST": datasets.KMNIST,
    "CIFAR10": datasets.CIFAR10,
}


def get_full_dataset(name=DATASET, root=DATA_ROOT):
    dataset_class = DATASETS[name]
    transform = transforms.ToTensor()
    train = dataset_class(root=str(root), train=True, download=True, transform=transform)
    test = dataset_class(root=str(root), train=False, download=True, transform=transform)
    return ConcatDataset([train, test])


def make_split_indices(n, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": permutation[:n_train].tolist(),
        "val": permutation[n_train:n_train + n_val].tolist(),
        "test": permutation[n_train + n_val:].tolist(),
    }


def get_split_indices(n, split_file=SPLIT_FILE, seed=SEED):
    try:
        indices = load_json(split_file)
        print(f"[split] dimuat dari {split_file}")
    except FileNotFoundError:
        indices = make_split_indices(n, seed=seed)
        save_json(indices, split_file)
        print(f"[split] dibuat baru dan disimpan ke {split_file}")
    return indices


def get_datasets(name=DATASET, root=DATA_ROOT, split_file=SPLIT_FILE):
    full = get_full_dataset(name, root)
    indices = get_split_indices(len(full), split_file)
    return {key: Subset(full, values) for key, values in indices.items()}


def get_loaders(batch_size=BATCH_SIZE, name=DATASET, root=DATA_ROOT, split_file=SPLIT_FILE):
    datasets_split = get_datasets(name, root, split_file)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return {
        "train": DataLoader(datasets_split["train"], batch_size=batch_size,
                            shuffle=True, num_workers=NUM_WORKERS, generator=generator),
        "val": DataLoader(datasets_split["val"], batch_size=batch_size,
                          shuffle=False, num_workers=NUM_WORKERS),
        "test": DataLoader(datasets_split["test"], batch_size=batch_size,
                           shuffle=False, num_workers=NUM_WORKERS),
    }


def to_minus1_1(images):
    return images * 2.0 - 1.0


def from_minus1_1(images):
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
