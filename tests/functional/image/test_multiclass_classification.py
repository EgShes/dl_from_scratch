from dl_from_scratch.examples.image.multiclass_classification import train_model


def test_learns():

    batch_size = 50
    num_epochs = 2
    learning_rate = 0.01

    train_metrics, val_metrics = train_model(batch_size, num_epochs, learning_rate)

    for metrics in [train_metrics, val_metrics]:
        assert metrics[0].loss - metrics[-1].loss > 1.0
        assert metrics[-1].auc - metrics[0].auc > 0.40
