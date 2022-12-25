from itertools import count
from typing import Sequence

import numpy as np
from numpy import ndarray

DataType = tuple[ndarray, ...]
DatasetType = Sequence[DataType]


class Loader:
    def __init__(self, dataset: DatasetType, batch_size: int, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        length = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size != 0:
            length += 1
        return length

    def __iter__(self):
        self._counter = count()
        self._indexes = np.arange(len(self.dataset))
        self._finished = False
        if self.shuffle:
            np.random.shuffle(self._indexes)
        return self

    def __next__(self) -> DataType:
        items = []

        if self._finished:
            raise StopIteration

        for _ in range(self.batch_size):
            index = next(self._counter)
            try:
                index = self._indexes[index]
            except IndexError:
                self._finished = True
                break
            items.append(self.dataset[index])

        if len(items) == 0:
            raise StopIteration

        return tuple(np.stack(kek) for kek in (elem for elem in zip(*items)))
