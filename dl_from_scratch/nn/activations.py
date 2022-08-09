from dl_from_scratch.nn.layer import Layer

import numpy as np


class ReLU(Layer):

    def __init__(self) -> None:
        pass

    def initialize(self):
        self._parameters = {}
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs >= 0, inputs, 0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(grad >= 0, 1, 0)
