from dl_from_scratch.nn.base import Loss

import numpy as np


class MSELoss(Loss):
    """Mean squared error layer
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
        self._data['inputs'] = inputs.copy()
        self._data['targets'] = targets.copy()
        loss = np.power(targets - inputs, 2)
        return loss

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass
        grad: any
        grad: (bs, out_channels)
        """
        grad = -2 * (self._data['targets'] - self._data['inputs'])
        return grad
        
    def __str__(self) -> str:
        return f'Sum: ()'
