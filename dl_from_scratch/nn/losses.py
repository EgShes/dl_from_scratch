import numpy as np

from dl_from_scratch.nn.base import Loss
from dl_from_scratch.nn.functions import softmax


class MSELoss(Loss):
    """Mean squared error loss
    l = (targets - inputs) ** 2

    dLdINPUTS = -2 * (targets - inputs)
    """

    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self._data = {}

    def zero_grad(self):
        pass

    def forward(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Forward pass
        O = (I + T) ** 2
        I: (bs, out_channels)
        T: (bs, out_channels)
        O: (bs, out_channels)
        """
        assert inputs.shape == targets.shape
        self._data["inputs"] = inputs.copy()
        self._data["targets"] = targets.copy()
        loss = np.power(targets - inputs, 2)
        return loss.mean()

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass
        grad: any
        grad: (bs, out_channels)
        """
        grad = -2 * (self._data["targets"] - self._data["inputs"])
        return grad

    def __str__(self) -> str:
        return "MSELoss()"


class CrossEntropyLoss(Loss):
    """Cross entropy loss
    l = (targets - inputs) ** 2

    dLdINPUTS = -2 * (targets - inputs)
    """

    def __init__(self, eps=1e-12) -> None:
        self.initialize()
        self._eps = eps

    def initialize(self):
        self._data = {}

    def zero_grad(self):
        pass

    def forward(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Forward pass
        P = softmax(X)
        O = 1/n * (-Y0*log(P0) - (1-Y0)*log(1-P0)) + ... + (-Yn*log(Pn) - (1-Yn)*log(1-Pn))
        O = 1/n * (-Y*log(P) - (1-Y)*log(1-P))
        X: (bs, in_channels)
        Y: (bs, in_channels)
        O: (bs, 1)
        """
        assert inputs.shape == targets.shape
        probs = softmax(inputs, dim=1)
        probs = np.clip(probs, self._eps, 1 - self._eps)  # make sure there is no 0 prob here

        self._data["probs"] = probs.copy()
        self._data["targets"] = targets.copy()
        loss = np.mean(-targets * np.log(probs) - (1 - targets) * np.log(1 - probs))
        return loss

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass
        grad = P - Y
        grad: any
        grad: (bs, in_channels)
        """
        grad = self._data["probs"] - self._data["targets"]
        return grad

    def __str__(self) -> str:
        return "CrossEntropyLoss()"
