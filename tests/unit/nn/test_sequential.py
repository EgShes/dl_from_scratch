import numpy as np
import pytest

from dl_from_scratch.nn import Linear, Sequential


class TestSequential:
    @pytest.fixture
    def forward_inputs(self) -> np.ndarray:
        return np.random.randn(2, 3)  # (batch, hidden1_in)

    @pytest.fixture
    def backward_inputs(self) -> np.ndarray:
        return np.random.randn(2, 10)  # (batch, hidden2_out)

    @pytest.fixture
    def model(self) -> Sequential:
        return Sequential(Linear(3, 20), Linear(20, 10))

    def test_forward(self, forward_inputs: np.ndarray, model: Sequential):
        logits = model(forward_inputs)
        assert logits.shape == (2, 10)

    def test_backward(
        self, forward_inputs: np.ndarray, backward_inputs: np.ndarray, model: Sequential
    ):
        _ = model(forward_inputs)
        grad = model.backward(backward_inputs)
        assert grad.shape == (2, 3)
