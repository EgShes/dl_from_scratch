import numpy as np

import dl_from_scratch.nn as nn
from dl_from_scratch.nn.optimization.base import Optimizer


class SGD(Optimizer):
    def step(self):
        for param in self.model.parameters():
            param.weight -= self.lr * param.gradient


class SGDMomentum(Optimizer):
    def __init__(self, model: nn.Sequential, learning_rate: float, momentum: float = 0.9):
        super().__init__(model, learning_rate)
        self.momentum = momentum
        self.velocity = None

    def step(self):
        if not self.velocity:
            self.velocity = [np.zeros_like(param.gradient) for param in self.model.parameters()]

        for param, velocity in zip(self.model.parameters(), self.velocity):
            velocity *= self.momentum
            velocity += self.lr * param.gradient

            param.weight -= velocity
