import torch

def reflect(v, n):
    """Reflects a vector v over a normal n."""
    return v - 2 * (v @ n) * n


def distance(x, y):
    """Returns the distance between two points."""
    return torch.norm(x - y, dim=-1)