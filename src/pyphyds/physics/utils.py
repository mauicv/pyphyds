import torch

def reflect(v, n):
    """Reflects a vector v over a normal n."""
    return v - 2 * (v @ n) * n


def distance(x, y):
    """Returns the distance between two points."""
    return torch.norm(x - y, dim=-1)


def equi_points(n):
    """Returns n equidistant and equiangular points in a circle radius with a
    random orientation."""
    angle = torch.tensor([i * (2 * 3.1415 / n) for i in range(n)])
    angle_perturb = torch.rand(n) * (2 * 3.1415)
    x = torch.cos(angle + angle_perturb)
    y = torch.sin(angle + angle_perturb)
    return torch.stack([x, y], dim=-1)