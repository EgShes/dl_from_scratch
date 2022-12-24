import logging

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import dl_from_scratch.nn as nn
from dl_from_scratch.data.datasets import get_mnist_loaders
from dl_from_scratch.nn.functions import softmax
from dl_from_scratch.nn.losses import CrossEntropyLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model: nn.Sequential, loader, loss: nn.losses.Loss) -> tuple[float, float]:
    preds, gts, losses = [], [], []
    for images, labels in tqdm(loader, total=len(loader), desc="Training"):
        # TODO build my own Dataset and DataLoader
        images, labels = images.numpy(), labels.numpy()

        images = images.reshape(batch_size, -1)
        # TODO make to_one_hot function
        labels = np.eye(10)[labels]

        model.zero_grad()
        pred = model(images)
        loss_val = loss(pred, labels)
        model.backward(loss.backward(1))

        preds.append(softmax(pred))
        gts.append(labels)
        losses.append(loss_val)

        # update weights
        for layer in model._layers:
            for name, param in layer.parameters.items():
                param.weight -= lr * param.gradient

    loss = np.mean(losses)

    preds, gts = np.concatenate(preds), np.concatenate(gts)
    auc = roc_auc_score(gts, preds)
    return loss, auc


def eval(
    model: nn.Sequential,
    loader,
    loss: nn.losses.Loss,
) -> tuple[float, float]:
    preds, gts, losses = [], [], []
    for images, labels in tqdm(loader, total=len(loader), desc="Evaluating"):
        # TODO build my own Dataset and DataLoader
        images, labels = images.numpy(), labels.numpy()

        images = images.reshape(1, -1)
        # TODO make to_one_hot function
        labels = np.eye(10)[labels]

        pred = model(images)
        loss_val = loss(pred, labels)

        preds.append(softmax(pred))
        gts.append(labels)
        losses.append(loss_val)

    loss = np.mean(losses)

    preds, gts = np.concatenate(preds), np.concatenate(gts)
    auc = roc_auc_score(gts, preds)
    return loss, auc


if __name__ == "__main__":

    batch_size = 32
    num_epochs = 5
    lr = 0.001

    train_loader, val_loader = get_mnist_loaders(batch_size=batch_size)

    model = nn.Sequential(
        nn.Linear(784, 300),
        nn.Sigmoid(),
        nn.Linear(300, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10),
    )

    loss = CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss, train_auc = train(model, train_loader, loss)
        logger.info(f"{train_loss=:.4f}, {train_auc=:.4f}")

        val_loss, val_auc = eval(model, val_loader, loss)
        logger.info(f"{val_loss=:.4f}, {val_auc=:.4f}")
