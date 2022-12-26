from dl_from_scratch.nn.optimization.base import Optimizer


class SGD(Optimizer):
    def step(self):
        for param in self.model.parameters():
            param.weight -= self.lr * param.gradient
