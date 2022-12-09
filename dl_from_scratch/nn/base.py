from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Parameter:
    weight: np.ndarray
    gradient: np.ndarray | None = None


class DifferentiableUnit(ABC):
    """Base class for every differentiable unit"""

    @abstractmethod
    def zero_grad(self):
        """Sets all gradients to zero and removes everything saved from the forward pass"""

    @abstractmethod
    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Performs a forward pass. Saves data needed for backward pass"""

    @abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        """Performs a backward pass"""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Parameter]:
        """Returns dict with parameters of the layer"""

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """The same as the forward pass"""
        return self.forward(*args, **kwargs)


class Layer(DifferentiableUnit):
    """Base class for every neural net layer"""

    def __init__(self, *args, **kwargs) -> None:
        """Creates and initializes parameters of a layer"""
        self._initialize_parameters()
        self._forward_info = {}

    def _initialize_parameters(self):
        """Initializes layer's weights"""
        self._parameters: dict[str, Parameter] = {}

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


class Activation(DifferentiableUnit):
    """Base class for every activation"""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Performs a forward pass. Saves data needed for backward pass"""

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs a backward pass"""

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Returns dict with parameters of the layer"""
        return {}


class Loss(DifferentiableUnit):
    """Base class for every loss"""

    @abstractmethod
    def forward(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Performs a forward pass. Saves data needed for backward pass"""

    @abstractmethod
    def backward(self) -> np.ndarray:
        """Performs a backward pass"""

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Returns dict with parameters of the layer"""
        return {}
