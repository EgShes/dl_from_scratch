from abc import ABC, abstractmethod

import dl_from_scratch.nn as nn


class Optimizer(ABC):
    def __init__(self, model: nn.Sequential, learning_rate: float, **kwargs):
        self.model = model
        self.lr = learning_rate

    @abstractmethod
    def step(self):
        """Updates model weights"""
