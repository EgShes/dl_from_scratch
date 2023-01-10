import pytest

from dl_from_scratch.nn.optimization.base import Optimizer
from dl_from_scratch.nn.scheduling.decay import LinearDecayScheduler


class FakeOptimizer(Optimizer):
    def __init__(self, model: any, learning_rate: float):
        super().__init__(model, learning_rate)

    def step(self):
        pass


class TestLinearDecay:
    @pytest.fixture
    def optimizer(self) -> FakeOptimizer:
        return FakeOptimizer(model=None, learning_rate=0.01)

    def test_lr_reset(self, optimizer: Optimizer):
        assert optimizer.lr == pytest.approx(0.01)
        _ = LinearDecayScheduler(optimizer, 0.05, 0, 10)
        assert optimizer.lr == pytest.approx(0.05)

    def test_lr_changes(self, optimizer: Optimizer):
        num_epochs = 10
        scheduler = LinearDecayScheduler(optimizer, 0.05, 0, num_epochs)
        lrs = []
        for _ in range(num_epochs):
            lrs.append(optimizer.lr)
            scheduler.step()

        diffs = []
        for i in range(1, len(lrs)):
            diffs.append(lrs[i - 1] - lrs[i])

        assert all([diff > 0 for diff in diffs])
        assert all([diff == pytest.approx(diffs[0]) for diff in diffs])
