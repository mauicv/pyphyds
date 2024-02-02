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
    pm = ParticleMap(p, 2, [0.5, 0.5], properties={1: {'size': 1}, 2: {'size': 0.5}})
    index = torch.tensor([1, 2, 2, 2, 1, 2, 1, 2, 1, 2])
    pm.particle_index = index
    sizes = pm.get_prop_vect('size')
    target = torch.zeros_like(index).to(torch.float32)
    target[index == 1] = 1
    target[index == 2] = 0.5
    assert (sizes == target).all()

