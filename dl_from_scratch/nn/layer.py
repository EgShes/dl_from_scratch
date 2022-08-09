from abc import ABC, abstractmethod

from dataclasses import dataclass

import numpy as np


@dataclass
class Parameter:
    weight: np.ndarray
    gradient: np.ndarray | None = None


class Layer(ABC):

    @abstractmethod
    def __init__(self) -> None:
        """Creates and initializes parameters of a layer"""

    @abstractmethod
    def initialize(self):
        """Initializes layer's weights"""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs a forward pass. Saves data needed for backward pass"""

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs a backward pass"""

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Returns dict with parameters of the layer"""
        return self._parameters

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """The same as the forward pass"""
        return self.forward(inputs)
