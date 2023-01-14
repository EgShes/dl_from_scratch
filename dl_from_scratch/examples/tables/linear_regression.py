import logging
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

import dl_from_scratch.nn as nn
from dl_from_scratch.data.loader import Loader
from dl_from_scratch.data.utils import get_boston_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metrics(NamedTuple):
    loss: float


def train_epoch(
    model: nn.Sequential, loader: Loader, loss: nn.losses.Loss, optimizer: nn.Optimizer
) -> Metrics:
    model.train()
    preds, gts, losses = [], [], []
    for inputs, labels in tqdm(loader, total=len(loader), desc="Training"):

        model.zero_grad()
        pred = model(inputs)
        loss_val = loss(pred, labels)
        model.backward(loss.backward(1))

        optimizer.step()

        preds.append(pred)
        gts.append(labels)
        losses.append(loss_val)

    loss = np.mean(losses)

    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return Metrics(loss)


def eval_epoch(
    model: nn.Sequential,
    loader: Loader,
    loss: nn.losses.Loss,
) -> Metrics:
    model.eval()
    preds, gts, losses = [], [], []
    for inputs, labels in tqdm(loader, total=len(loader), desc="Evaluating"):
        pred = model(inputs)
        loss_val = loss(pred, labels)

        preds.append(pred)
        gts.append(labels)
        losses.append(loss_val)

    loss = np.mean(losses)

    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return Metrics(loss)


def train_model(
    batch_size: int = 50,
    num_epochs: int = 5,
    lr: float = 0.0001,
) -> tuple[list[Metrics], list[Metrics]]:
    train_metrics, val_metrics = [], []

    train_loader, val_loader = get_boston_loaders(batch_size=batch_size)

    model = nn.Sequential(
        nn.Linear(13, 7),
        nn.Linear(7, 1),
    )

    loss = nn.MSELoss()
    optimizer = nn.SGD(model=model, learning_rate=lr)

    logger.info("Evaluating model before training")
    train_metrics.append(eval_epoch(model, train_loader, loss))
    val_metrics.append(eval_epoch(model, val_loader, loss))

    logger.info("Start training")
    for epoch in range(num_epochs):
        train_metric = train_epoch(model, train_loader, loss, optimizer)
        logger.info(f"Epoch {epoch+1}: {train_metric.loss=:.4f}")

        val_metric = eval_epoch(model, val_loader, loss)
        logger.info(f"Epoch {epoch+1}: {val_metric.loss=:.4f}")

        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

    return train_metrics, val_metrics


if __name__ == "__main__":

    batch_size = 5
    num_epochs = 5
    learning_rate = 0.001

    train_model(batch_size, num_epochs, learning_rate)
