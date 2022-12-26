from pathlib import Path
from typing import Tuple

import torchvision
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl_from_scratch.data.dataset import NumpyDataset
from dl_from_scratch.data.loader import Loader

Loaders = Tuple[Loader, Loader]  # train, val loaders

# TODO make it work on windows
DATASET_DIR = Path("/tmp")


def get_boston_loaders(batch_size: int) -> Loaders:
    boston = load_boston()
    data, target = boston.data, boston.target
    s = StandardScaler()
    data = s.fit_transform(data)

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.3, random_state=1)
    y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)

    train_dataset = NumpyDataset(x_train, y_train)
    val_dataset = NumpyDataset(x_val, y_val)

    train_loader = Loader(train_dataset, batch_size, shuffle=True)
    val_loader = Loader(val_dataset, 1, shuffle=False)

    return train_loader, val_loader


def get_mnist_loaders(batch_size: int) -> Loaders:
    common_params = {
        "root": DATASET_DIR,
        "download": True,
        "transform": torchvision.transforms.ToTensor(),
    }

    train = torchvision.datasets.MNIST(train=True, **common_params)
    val = torchvision.datasets.MNIST(train=False, **common_params)

    train_loader = Loader(train, batch_size, shuffle=True)
    val_loader = Loader(val, 1, shuffle=False)

    return train_loader, val_loader
