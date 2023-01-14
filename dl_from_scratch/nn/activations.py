import numpy as np

from dl_from_scratch.nn.base import Layer


class ReLU(Layer):
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

    def __repr__(self) -> str:
        return "ReLU: ()"


class Sigmoid(Layer):
    """
    forward: 1 / (1 + np.exp(-inputs)) == sigmoid(inputs)
    backward: sigmoid(inputs) * (1 - sigmoid(inputs))
    """

    def __init__(self) -> None:
        super().__init__()
        self._forward_info = {}

    def zero_grad(self) -> None:
        self._forward_info = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        in: (bs, feature_size)
        out: (bs, feature_size)
        """
        sigmoid = 1 / (1 + np.exp(-inputs))
        self._forward_info["sigmoid"] = sigmoid.copy()
        return sigmoid

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        in: (bs, feature_size)
        out: (bs, feature_size)
        """
        return grad * self._forward_info["sigmoid"] * (1 - self._forward_info["sigmoid"])

    def __repr__(self) -> str:
        return "Sigmoid: ()"
