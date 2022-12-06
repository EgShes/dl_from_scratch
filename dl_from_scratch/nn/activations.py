from dl_from_scratch.nn.base import Activation

import numpy as np


class ReLU(Activation):
    """
    forward: x if x >= 0 else 0
    backward: 1 if grad >= 0 else 0
    """

    def zero_grad(self) -> None:
        return None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs >= 0, inputs, 0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(grad >= 0, 1, 0)


class Sigmoid(Activation):
    """
    forward: 1 / (1 + np.exp(-inputs)) == sigmoid(inputs)
    backward: sigmoid(inputs) * (1 - sigmoid(inputs))
    """

    def __init__(self) -> None:
        self._forward_info = {}

    def zero_grad(self) -> None:
        self._forward_info = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        in: (bs, feature_size)
        out: (bs, feature_size)
        """
        sigmoid = 1 / (1 + np.exp(-inputs))
        self._forward_info['sigmoid'] = sigmoid.copy()
        return sigmoid

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        in: (bs, feature_size)
        out: (bs, feature_size)
        """
        return grad * self._forward_info['sigmoid'] * (1 - self._forward_info['sigmoid'])
