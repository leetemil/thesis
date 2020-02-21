import math
import time
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from utils import make_loading_bar, readable_time, eta, get_gradient_norm

def log_progress(epoch, time, progress, total, end, **kwargs):
    report = f"Epoch: {epoch:5} "
    digits = int(math.log10(total)) + 1
    report += f"Time: {readable_time(time)} ETA: {readable_time(eta(time, progress / total))} [{progress:{digits}}/{total}] {make_loading_bar(40, progress / total)}"

    for key, value in kwargs.items():
        if type(value) == int:
            report += (f" {key}: {value:5}")
        elif type(value) == float:
            report += (f" {key}: {value:7.5f}")
        else:
            report += (f" {key}: {value}")
    print(report, end = end)

def train_epoch(epoch, model, optimizer, train_loader, log_interval, clip_grad_norm = None, clip_grad_value = None):
    """
        epoch: Index of the epoch to run
        model: The model to run data through. Forward should return a tuple of (loss, metrics_dict).
        optimizer: The optimizer to step with at every batch
        train_loader: PyTorch DataLoader to generate batches of training data
        log_interval: Interval in seconds of how often to log training progress (0 to disable batch progress logging)
    """
    progressed_data = 0
    data_len = len(train_loader.dataset)
    num_batches = (data_len // train_loader.batch_size) + 1
    if log_interval != 0:
        log_progress(epoch, 0, progressed_data, data_len, "\r", Loss = 0)
    last_log_time = time.time()

    train_loss = 0
    train_count = 0
    start_time = time.time()

    acc_metrics_dict = defaultdict(lambda: 0)
    for batch_idx, xb in enumerate(train_loader):
        batch_size, loss, batch_metrics_dict = train_batch(model, optimizer, xb, clip_grad_norm, clip_grad_value)

        progressed_data += batch_size

        # Calculate accumulated metrics
        for key, value in batch_metrics_dict.items():
            acc_metrics_dict[key] += value
            acc_metrics_dict[key + "_count"] += 1
        metrics_dict = {k: acc_metrics_dict[k] / acc_metrics_dict[k + "_count"] for k in acc_metrics_dict.keys() if not k.endswith("_count")}

        train_loss += loss
        train_count += 1

        if log_interval != 0 and (log_interval == "batch" or time.time() - last_log_time > log_interval):
            last_log_time = time.time()
            log_progress(epoch, time.time() - start_time, progressed_data, data_len, "\r", Loss = train_loss / train_count, **metrics_dict)

    average_loss = train_loss / train_count
    if log_interval != 0:
        log_progress(epoch, time.time() - start_time, data_len, data_len, "\n", Loss = train_loss / train_count, **metrics_dict)
    return average_loss

def train_batch(model, optimizer, xb, clip_grad_norm = None, clip_grad_value = None):
    model.train()
    batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)

    # Reset gradient for next batch
    optimizer.zero_grad()

    # Push whole batch of data through model.forward()
    if isinstance(xb, Tensor):
        loss, batch_metrics_dict = model(xb)
    else:
        loss, batch_metrics_dict = model(*xb)

    # Calculate the gradient of the loss w.r.t. the graph leaves
    loss.backward()

    if clip_grad_norm is not None:
        clip_grad_norm_(model.parameters(), clip_grad_norm)
    if clip_grad_value is not None:
        clip_grad_value_(model.parameters(), clip_grad_value)

    # Step in the direction of the gradient
    optimizer.step()

    loss_return = loss.item()

    # Last usage of loss above: Delete it
    del loss

    return batch_size, loss_return, batch_metrics_dict

def validate(epoch, model, validation_loader):
    model.eval()

    validation_loss = 0
    validation_count = 0
    with torch.no_grad():
        for i, xb in enumerate(validation_loader):
            batch_size = xb.size(0) if isinstance(xb, torch.Tensor) else xb[0].size(0)

            # Push whole batch of data through model.forward()
            if isinstance(xb, Tensor):
                loss, batch_metrics_dict = model(xb)
            else:
                loss, batch_metrics_dict = model(*xb)

            validation_loss += loss.item()
            validation_count += 1

        average_loss = validation_loss / validation_count
    return average_loss
