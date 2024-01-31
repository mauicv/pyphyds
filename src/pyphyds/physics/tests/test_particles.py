import torch
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap


def test_particles():
    p = Particles(1, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    x = torch.tensor([5., 5.])
    v = torch.tensor([0.1, 0.1])
    p.x = torch.clone(x)
    p.v = v
    p.step()
    assert (p.x == x + v).all()

def test_particle_map():
    p = Particles(10, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    pm = ParticleMap(p, [0, 1], [0.5, 0.5])
    a = pm.get_class('x', 1)
    pm.set_class('x', 1, torch.zeros_like(a))
