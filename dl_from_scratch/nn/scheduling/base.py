from abc import ABC, abstractmethod

from dl_from_scratch.nn.optimization.base import Optimizer


class Scheduler(ABC):
    def __init__(self, optimizer: Optimizer, **kwargs) -> None:
        self.optimizer = optimizer

    @abstractmethod
    def step(self):
        """Updates learning rate of the optimizer"""
