import pytest

from dl_from_scratch.nn.linear import MSE, Linear
from dl_from_scratch.nn.sequential import Sequential


@pytest.mark.parametrize(
    'layers',
    [
        (Linear(13, 1), ),
        (Linear(13, 7), Linear(7, 1)),
    ]
)
def test_learns(boston_train, boston_val, layers):
    model = Sequential(*layers)
    loss = MSE()

    num_epochs = 50
    lr = 0.0001

    losses = []
    for _ in range(num_epochs):

        for x, y in boston_train:
            pred = model(x)
            loss_value = loss.forward(pred, y)

            # backward
            model.backward(loss.backward(1))

            # update weights
            for layer in model._layers:
                for name, param in layer.parameters.items():
                    param.weight -= lr * param.gradient

            # zero grad
            model.zero_grad()

            losses.append(loss_value.mean())

    max_loss, min_loss = max(losses), min(losses)
    assert min_loss / max_loss < 0.05
