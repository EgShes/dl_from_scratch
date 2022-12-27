from dl_from_scratch.nn.optimization.base import Optimizer
from dl_from_scratch.nn.scheduling.base import Scheduler


class LinearDecayScheduler(Scheduler):
    def __init__(
        self, optimizer: Optimizer, initial_lr: float, final_lr: float, num_epochs: int
    ) -> None:
        super().__init__(optimizer)
        assert initial_lr >= final_lr

        self._initial_lr = initial_lr
        self._final_lr = final_lr
        self._delta = (initial_lr - final_lr) / num_epochs

        self.optimizer.lr = self._initial_lr

    def step(self):
        new_lr = self.optimizer.lr - self._delta
        if new_lr < self._final_lr:
            new_lr = self._final_lr

        self.optimizer.lr = new_lr
