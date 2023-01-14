import numpy as np

from dl_from_scratch.nn.base import Layer


class Dropout(Layer):
    """Dropout layer
    Randomly zero some fraction of neurons
    """

    def __init__(
        self,
        probability: float,
    ) -> None:
        super().__init__()
        self._probability = probability

    def zero_grad(self) -> None:
        return None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass
        X: (bs, channels)
        O: (bs, channels)
        """
        if self._train_mode:
            self._forward_info["mask"] = np.random.binomial(
                1, (1 - self._probability), size=inputs.shape
            )
            output = inputs * self._forward_info["mask"]
        else:
            output = inputs * (1 - self._probability)
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass
        grad_in: (bs, channels)
        grad_out: (bs, channels)
        """
        # (bs, channels) * (bs, channels) -> (bs, channels)
        grad = grad * self._forward_info["mask"]
        return grad

    def __repr__(self) -> str:
        return f"Dropout: (p={self._probability})"
