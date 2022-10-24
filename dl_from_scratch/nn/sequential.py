from typing import Any, Iterable

from dl_from_scratch.nn.base import Layer
import numpy as np


class Sequential:

    def __init__(self, *layers: Layer) -> None:
        self._layers = list(layers)

    def initialize(self):
        return

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in self._layers[::-1]:
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)
