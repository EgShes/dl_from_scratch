from dl_from_scratch.nn.layer import Layer, Parameter

import numpy as np


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

        self._parameters['w'] = Parameter(np.random.randn(self._in_channels, self._out_channels))
        self._parameters['b'] = Parameter(np.zeros((1, out_channels)))

        self._data = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._data['x'] = inputs.copy()
        return np.dot(inputs, self._parameters['w'].weight) + self._parameters['b'].weight

    def backward(self, grad: np.ndarray) -> None:
        self._parameters['w'].gradient = np.dot(self._data['x'], grad)
        self._parameters['b'].gradient = np.dot(np.ones(1, self._out_channels), grad)

    def __str__(self) -> str:
        return f'Linear: ({self._in_channels, self._out_channels})'


if __name__ == '__main__':
    in_channels = 10
    out_channels = 20
    
    input = np.random.randn(15, in_channels)
    layer = Linear(in_channels, out_channels)
    pred = layer(input)
    print(pred.shape)

    gradient = np.random.randn(out_channels, 7)
    layer.backward(gradient)
