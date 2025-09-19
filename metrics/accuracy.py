
import torch

def get_accuracy(model, X, y):
    with torch.no_grad():
        op = model(X)
        accuracy = (torch.argmax(op, dim=1) == y).float().mean().item()
    return accuracy