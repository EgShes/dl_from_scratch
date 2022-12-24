from pathlib import Path
from typing import Tuple

import torchvision
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# from dl_from_scratch.data.utils import Loader


Loaders = Tuple[DataLoader, DataLoader]  # train, val loaders

# TODO make it work on windows
DATASET_DIR = Path("/tmp")


def get_boston_loaders(batch_size: int) -> Loaders:
    boston = load_boston()
    data, target = boston.data, boston.target
    s = StandardScaler()
    data = s.fit_transform(data)

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.3, random_state=1)
    y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)

    train_loader = DataLoader(x_train, y_train, batch_size, shuffle=True)
    val_loader = DataLoader(x_val, y_val, 1, shuffle=True)

    return train_loader, val_loader


def get_mnist_loaders(batch_size: int) -> Loaders:
    common_params = {
        "root": DATASET_DIR,
        "download": True,
        "transform": torchvision.transforms.ToTensor(),
    }
    # x_train, y_train = zip(*torchvision.datasets.MNIST(train=True, **common_params))
    # x_val, y_val = zip(*torchvision.datasets.MNIST(train=False, **common_params))

    train = torchvision.datasets.MNIST(train=True, **common_params)
    val = torchvision.datasets.MNIST(train=False, **common_params)

    train_loader = DataLoader(train, batch_size, shuffle=True)
    val_loader = DataLoader(val, 1, shuffle=True)

    return train_loader, val_loader


if __name__ == "__main__":
    t, v = get_mnist_loaders(4)
    t_b = next(iter(t))
    v_b = next(iter(v))
    a = 1
