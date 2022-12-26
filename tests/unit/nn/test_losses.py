import numpy as np
import pytest

from dl_from_scratch.nn.losses import CrossEntropyLoss


class TestCrossEntropyLoss:
    @pytest.fixture
    def ce_loss(self) -> CrossEntropyLoss:
        return CrossEntropyLoss()

    @pytest.mark.parametrize("ndim", list(range(2, 10)))
    def test_forward(self, ce_loss: CrossEntropyLoss, ndim: int):
        dims = [5] * ndim
        inputs, targets = np.random.randn(*dims), np.random.randint(0, 2, dims)
        loss = ce_loss(inputs, targets)
        assert isinstance(loss, float)

    @pytest.mark.parametrize("ndim", list(range(2, 10)))
    def test_backward(self, ce_loss: CrossEntropyLoss, ndim: int):
        dims = [5] * ndim
        inputs, targets = np.random.randn(*dims), np.random.randint(0, 2, dims)
        _ = ce_loss(inputs, targets)
        grad = ce_loss.backward(1)
        assert grad.shape == inputs.shape
        assert grad.shape == targets.shape

    def test_shapes_differ(self, ce_loss: CrossEntropyLoss):
        with pytest.raises(AssertionError):
            inputs, targets = np.random.randn(*[5, 5, 5]), np.random.randint(0, 2, [5, 5])
            ce_loss(inputs, targets)

    def test_zero_prob(self, ce_loss: CrossEntropyLoss):
        inputs = np.array(
            [
                [0.0, 0.1, 500.0],
                [0.4, 0.0, 500.0],
            ]
        )
        targets = np.random.randint(0, 2, [2, 3])
        ce_loss(inputs, targets)
