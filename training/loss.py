import torch
import torch.nn.functional as F

def orthogonality_loss(h):
    """
    h: (batch_size, hidden_dim) activations
    returns scalar regularization loss
    """
    h_norm = F.normalize(h, p=2, dim=1)  # shape: (batch_size, hidden_dim)

    G = torch.matmul(h_norm, h_norm.T)  # shape: (batch_size, batch_size)

    I = torch.eye(G.size(0))
    off_diag = G * (1 - I)

    loss = (off_diag**2).mean()
    return loss