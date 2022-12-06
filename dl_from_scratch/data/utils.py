from typing import Tuple

import numpy as np
from numpy import ndarray


class Loader:

    def __init__(self, x_data: ndarray, y_data: ndarray, batch_size: int, shuffle: bool = False):
        self._x_data = x_data
        self._y_data = y_data
        self._batch_size = batch_size
        
        if shuffle:
            self._shuffle_data

    def _shuffle_data(self):
        perm = np.random.permutation(self._x_data.shape[0])
        self._x_data = self._x_data[perm]
        self._y_data = self._y_data[perm]

    def __iter__(self):
        self._length = self._x_data.shape[0]
        self._range = iter(range(0, self._length, self._batch_size))
        return self

    def __next__(self) -> Tuple[ndarray, ndarray]:
        i = next(self._range)
        start, end = i, i+self._batch_size
        if start > self._length:
            raise StopIteration
        return self._x_data[start:end], self._y_data[start:end]
        

# if __name__ == '__main__':

#     from sklearn.datasets import load_boston
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.model_selection import train_test_split
#     boston = load_boston()
#     data = boston.data
#     target = boston.target

#     s = StandardScaler()
#     data = s.fit_transform(data)

#     x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
#     y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

#     loader = Loader(x_train, y_train, 3)
#     for x, y in loader:
#         print(x.shape, y.shape)
