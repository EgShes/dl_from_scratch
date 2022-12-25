from typing import Sequence

from numpy import ndarray


class NumpyDataset(Sequence):
    def __init__(self, x_data: ndarray, y_data: ndarray) -> None:
        self.x_data = x_data
        self.y_data = y_data
        assert self.x_data.shape[0] == self.y_data.shape[0]

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, item: int) -> tuple[ndarray, ndarray]:
        return self.x_data[item], self.y_data[item]
