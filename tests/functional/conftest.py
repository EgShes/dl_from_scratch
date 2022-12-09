from typing import Tuple

import pytest
from numpy import ndarray
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl_from_scratch.data.utils import Loader

BATCH_SIZE = 25


@pytest.fixture(scope="session")
def boston_data() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    boston = load_boston()
    data, target = boston.data, boston.target
    s = StandardScaler()
    data = s.fit_transform(data)

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.3, random_state=1)
    y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)

    return x_train, x_val, y_train, y_val


@pytest.fixture(scope="session")
def boston_train(boston_data) -> Loader:
    x_train, _, y_train, _ = boston_data
    return Loader(x_train, y_train, BATCH_SIZE, shuffle=True)


@pytest.fixture(scope="session")
def boston_val(boston_data) -> Loader:
    _, x_val, _, y_val = boston_data
    return Loader(x_val, y_val, 1, shuffle=False)
