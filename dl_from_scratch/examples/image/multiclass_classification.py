import logging
from typing import NamedTuple

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import dl_from_scratch.nn as nn
from dl_from_scratch.data.loader import Loader
from dl_from_scratch.data.utils import get_mnist_loaders
from dl_from_scratch.nn.functions import softmax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metrics(NamedTuple):
    loss: float
    auc: float


def train_epoch(model: nn.Sequential, loader: Loader, loss: nn.losses.Loss, lr: float) -> Metrics:
    preds, gts, losses = [], [], []
    for images, labels in tqdm(loader, total=len(loader), desc="Training"):
        bs = images.shape[0]

        images = images.reshape(bs, -1)
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
    return Metrics(loss, auc)


def eval_epoch(
    model: nn.Sequential,
    loader: Loader,
    loss: nn.losses.Loss,
) -> Metrics:
    preds, gts, losses = [], [], []
    for images, labels in tqdm(loader, total=len(loader), desc="Evaluating"):
        bs = images.shape[0]

        images = images.reshape(bs, -1)
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
    return Metrics(loss, auc)


def train_model(
    batch_size: int = 50,
    num_epochs: int = 5,
    lr: float = 0.001,
) -> tuple[list[Metrics], list[Metrics]]:
    train_metrics, val_metrics = [], []

    train_loader, val_loader = get_mnist_loaders(batch_size=batch_size)

    model = nn.Sequential(
        nn.Linear(784, 300),
        nn.Sigmoid(),
        nn.Linear(300, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10),
    )

    loss = nn.CrossEntropyLoss()

    logger.info("Evaluating model before training")
    train_metrics.append(eval_epoch(model, train_loader, loss))
    val_metrics.append(eval_epoch(model, val_loader, loss))

    logger.info("Start training")
    for _ in range(num_epochs):
        train_metric = train_epoch(model, train_loader, loss, lr)
        logger.info(f"{train_metric.loss=:.4f}, {train_metric.auc=:.4f}")

        val_metric = eval_epoch(model, val_loader, loss)
        logger.info(f"{val_metric.loss=:.4f}, {val_metric.auc=:.4f}")

        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

    return train_metrics, val_metrics


if __name__ == "__main__":

    batch_size = 101
    num_epochs = 5
    learning_rate = 0.001

    train_model(batch_size, num_epochs, learning_rate)
