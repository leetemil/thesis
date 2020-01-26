import time

import torch

from utils import make_loading_bar, readable_time, eta

def log_progress(epoch, time, fraction_done, loss, end):
    print(f"Epoch: {epoch:4} Loss: {loss:7.4f} {make_loading_bar(50, fraction_done)} Time: {readable_time(time)} ETA: {readable_time(eta(time, fraction_done))}", end = end)

def train(epoch, model, optimizer, loss_function, train_loader, log_interval):
    """
        epoch: Index of the epoch to run
        model: The model to run data through
        optimizer: The optimizer to step with at every batch
        loss_function: The output of the model will be passed through the loss function, which will be minimized by the optimizer
        train_loader: PyTorch DataLoader to generate batches of training data
        log_interval: Interval in seconds of how often to log training progress (0 to disable batch progress logging)
    """
    model.train()

    if log_interval != 0:
        log_progress(epoch, 0, 0, 0, "\r")
    last_log_time = time.time()

    train_loss = 0
    train_count = 0
    start_time = time.time()
    for batch_idx, xb in enumerate(train_loader):
        batch_size, seq_len = xb.shape

        # Reset gradient for next batch
        optimizer.zero_grad()

        # Push whole batch of data through model.forward()
        output = model(xb)

        # Calculate scalar loss
        if type(output) == tuple or type(output) == list:
            loss = loss_function(*output)
        else:
            loss = loss_function(output)

        # Calculate the gradient of the loss w.r.t. the graph leaves
        loss.backward()

        # Step in the direction of the gradient
        optimizer.step()

        train_loss += loss.item()
        train_count += batch_size * seq_len

        if log_interval != 0 and time.time() - last_log_time > log_interval:
            last_log_time = time.time()
            log_progress(epoch, time.time() - start_time, (batch_idx + 1) / len(train_loader), train_loss / train_count, "\r")

    average_loss = train_loss / train_count
    if log_interval != 0:
        log_progress(epoch, time.time() - start_time, (batch_idx + 1) / len(train_loader), train_loss / train_count, "\n")
    return average_loss

def validate(epoch, model, loss_function, validation_loader):
    model.eval()

    validation_loss = 0
    validation_count = 0
    with torch.no_grad():
        for i, xb in enumerate(validation_loader):
            batch_size, seq_len = xb.shape

            # Push whole batch of data through model.forward()
            output = model(xb)

            # Calculate scalar loss
            if type(output) == tuple or type(output) == list:
                loss = loss_function(*output)
            else:
                loss = loss_function(output)

            validation_loss += loss.item()
            validation_count += batch_size * seq_len

        average_loss = validation_loss / validation_count
    return average_loss
