import torch

def smooth_one_hot(t, classes, smoothing = 0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    t_size = torch.Size((*t.shape, classes))
    with torch.no_grad():
        smoothed_one_hot = torch.empty(size = t_size, device = t.device)
        smoothed_one_hot.fill_(smoothing / (classes - 1))
        smoothed_one_hot.view(-1, classes).scatter_(1, t.flatten().unsqueeze(-1), confidence)
    return smoothed_one_hot
