import math
import time
from collections import defaultdict
from collections.abc import Iterable

import torch

from utils import make_loading_bar, readable_time, eta, get_gradient_norm

def log_progress(epoch, time, fraction_done, progress, total, end, **kwargs):
    report = f"Epoch: {epoch:5} "
    digits = int(math.log10(total)) + 1
    report += f"Time: {readable_time(time)} ETA: {readable_time(eta(time, fraction_done))} [{progress:{digits}}/{total}] {make_loading_bar(40, fraction_done)}"

    for key, value in kwargs.items():
        if type(value) == int:
            report += (f" {key}: {value:5}")
        elif type(value) == float:
            report += (f" {key}: {value:7.5f}")
        else:
            report += (f" {key}: {value}")
    print(report, end = end)

def train(epoch, model, optimizer, train_loader, log_interval):
    """
        epoch: Index of the epoch to run
        model: The model to run data through. Forward should return a tuple of (loss, metrics_dict).
        optimizer: The optimizer to step with at every batch
        train_loader: PyTorch DataLoader to generate batches of training data
        log_interval: Interval in seconds of how often to log training progress (0 to disable batch progress logging)
    """
    model.train()

    progressed_data = 0
    data_len = len(train_loader.dataset)
    if log_interval != 0:
        log_progress(epoch, 0, 0, progressed_data, data_len, "\r", Loss = 0)
    last_log_time = time.time()

    train_loss = 0
    train_count = 0
    start_time = time.time()

    acc_metrics_dict = defaultdict(lambda: 0)
    for batch_idx, xb in enumerate(train_loader):
        batch_size, seq_len = xb.shape if isinstance(xb, torch.Tensor) else xb[0].shape
        progressed_data += batch_size

        # Reset gradient for next batch
        optimizer.zero_grad()

        # Push whole batch of data through model.forward()
        loss, batch_metrics_dict = model(*xb if isinstance(xb, Iterable) else xb)

        # Calculate accumulated metrics
        for key, value in batch_metrics_dict.items():
            acc_metrics_dict[key] += value
            acc_metrics_dict[key + "_count"] += 1
        metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}

        # Calculate the gradient of the loss w.r.t. the graph leaves
        loss.backward()

        # Step in the direction of the gradient
        optimizer.step()

        train_loss += loss.item()
        train_count += batch_size * seq_len

        # Last usage of loss above: Delete it
        del loss

        if log_interval != 0 and (log_interval == "batch" or time.time() - last_log_time > log_interval):
            last_log_time = time.time()
            log_progress(epoch, time.time() - start_time, (batch_idx + 1) / len(train_loader), progressed_data, data_len, "\r", Loss = train_loss / train_count, **metrics_dict)

    average_loss = train_loss / train_count
    if log_interval != 0:
        log_progress(epoch, time.time() - start_time, 1.0, progressed_data, data_len, "\n", Loss = train_loss / train_count, **metrics_dict)
    return average_loss

def validate(epoch, model, validation_loader):
    model.eval()

    validation_loss = 0
    validation_count = 0
    with torch.no_grad():
        for i, (xb, weights) in enumerate(validation_loader):
            batch_size, seq_len = xb.shape

            # Push whole batch of data through model.forward()
            loss, metrics_dict = model(xb, weights)

            validation_loss += loss.item()
            validation_count += batch_size * seq_len

        average_loss = validation_loss / validation_count
    return average_loss
