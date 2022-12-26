from dl_from_scratch.examples.tables.linear_regression import train_model


def test_learns():

    batch_size = 5
    num_epochs = 5
    learning_rate = 0.001

    train_metrics, val_metrics = train_model(batch_size, num_epochs, learning_rate)

    for metrics in [train_metrics, val_metrics]:
        assert metrics[0].loss - metrics[-1].loss > 100
