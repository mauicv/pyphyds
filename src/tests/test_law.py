import torch
import pytest
from pyphyds.particles import Particles
from pyphyds.laws.box_boundary import BoxBoundaryLaw


@pytest.mark.parametrize('p,t', [
    ([[-0.9, 0.2]], [[-0.9, 0.2]]),
    ([[-1.1, 0.2]], [[-0.9, 0.2]]),
    ([[-0.9, 0.2], [-1.5, 0.2]], [[-0.9, 0.2], [-0.5, 0.2]]),
    ([[0.9, -0.2]], [[0.9, -0.2]]),
    ([[1.1, -0.2]], [[0.9, -0.2]]),
    ([[0.9, -0.5], [1.1, 0.2]], [[0.9, -0.5], [0.9, 0.2]]),
    ([[0.9, -2.2]], [[0.9, 0.2]]),
    ([[0.9, -0.2]], [[0.9, -0.2]]),
    ([[0.9, -1.5], [0.1, -0.5]], [[0.9, -0.5], [0.1, -0.5]]),
    ([[1.1, -0.4], [1.1, -0.2]], [[0.9, -0.4], [0.9, -0.2]]),
])
def test_box_boundary_law(p, t):
    particles = Particles(1)
    particles.x = torch.tensor(p)
    particles.old_x = torch.tensor(p)
    boundary_law = BoxBoundaryLaw(particles, -1.0, 1.0, -1.0, 1.0)
    boundary_law.step()
    assert torch.allclose(particles.x, torch.tensor(t), atol=1e-5)

def test_box_boundary_law_random_particles():
    particles = Particles(100)
    assert (particles.x[:, 0] > 1).sum() > 0
    assert (particles.x[:, 0] < -1).sum() > 0
    assert (particles.x[:, 1] > 1).sum() > 0
    assert (particles.x[:, 1] < -1).sum() > 0
    boundary_law = BoxBoundaryLaw(particles, -1.0, 1.0, -1.0, 1.0)
    boundary_law.step()
    assert (particles.x[:, 0] > 1).sum() == 0
    assert (particles.x[:, 0] < -1).sum() == 0
    assert (particles.x[:, 1] > 1).sum() == 0
    assert (particles.x[:, 1] < -1).sum() == 0