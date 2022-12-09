from typing import Optional
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def softmax(x: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=dim, keepdims=True)

from scipy.special import logsumexp
def softmax_kek(x, dim=None):
    return np.exp(x - logsumexp(x, axis=dim, keepdims=True))