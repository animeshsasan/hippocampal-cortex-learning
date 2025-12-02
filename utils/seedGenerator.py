import torch

def generate_seeds(master_seed: int, n: int):
    """
    Generate `n` deterministic random seeds based on a single master seed.
    """
    g = torch.Generator().manual_seed(master_seed)
    seeds = torch.randint(0, 2**32 - 1, (n,), generator=g).tolist()
    return seeds