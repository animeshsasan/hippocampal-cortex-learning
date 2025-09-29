
import torch

def get_accuracy(model, X, y):
    with torch.no_grad():
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
        op = model(X)
        accuracy = (torch.argmax(op, dim=1) == y).float().mean().item()
    return accuracy