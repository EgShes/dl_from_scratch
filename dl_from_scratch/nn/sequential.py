from turtle import forward
from typing import Iterable

from dl_from_scratch.nn.layer import Layer
import numpy as np


class Sequential:

    def __init__(self, *layers: Iterable[Layer]) -> None:
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
