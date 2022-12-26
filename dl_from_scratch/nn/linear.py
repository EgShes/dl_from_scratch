import numpy as np

from dl_from_scratch.nn.base import Layer, Parameter


class Linear(Layer):
    """Linear layer
    y = w * x + b

    y' (w) = x
    y' (b) = 1
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.initialize()

    def initialize(self):
        self._parameters: dict[str, Parameter] = {}

        self._parameters["w"] = Parameter(
            weight=np.random.randn(self._in_channels, self._out_channels),
            gradient=np.zeros((self._in_channels, self._out_channels)),
        )
        self._parameters["b"] = Parameter(
            weight=np.zeros((1, self._out_channels)),
            gradient=np.zeros((1, self._out_channels)),
        )

        self._forward_info = {}

    def zero_grad(self):
        self._parameters["w"].gradient = np.zeros((self._in_channels, self._out_channels))
        self._parameters["b"].gradient = np.zeros((1, self._out_channels))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass
        O = WX + B
        X: (bs, in_channels)
        W: (in_channels, out_channels)
        B: (1, out_channels)
        O: (bs, out_channels)
        """
        self._forward_info["x"] = inputs.copy()
        self._forward_info["w"] = self._parameters["w"].weight.copy()
        self._forward_info["b"] = self._parameters["b"].weight.copy()
        output = np.dot(inputs, self._parameters["w"].weight) + self._parameters["b"].weight
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass
        grad_in: (bs, out_channels)
        dLdW: (in_channels, out_channels)
        dLdB: (1, out_channels)
        dLdO: (bs, out_channels)
        grad_out: (bs, in_channels)
        """
        # (in_channels, bs) @ (bs, out_dim) -> (in_channels, out_dim)
        self._parameters["w"].gradient += np.dot(self._forward_info["x"].T, grad)

        # (1, out_channels) * (bs, out_channels) -> (bs, out_channels) -> (1, out_channels)
        self._parameters["b"].gradient += (np.ones_like(self._forward_info["b"]) * grad).sum(
            0, keepdims=True
        )

        # (bs, out_channels) @ (out_channels, in_channels) -> (bs, in_channels)
        grad = np.dot(grad, self._parameters["w"].weight.T)
        return grad

    def __str__(self) -> str:
        return f"Linear: ({self._in_channels, self._out_channels})"
