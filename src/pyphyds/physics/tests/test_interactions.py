import numpy as np
import torch
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction


def test_particles():
    p = Particles(2, x_bound=np.array([10., 10.]), v_bound=1)
    pm = ParticleMap(p, [0, 1], [0.5, 0.5], properties={0: {'size': 1}, 1: {'size': 1}})
    p.x = torch.tensor([[5., 5.], [5., 5.]])
    p.v = torch.tensor([[1., -1.], [-1., 0.]])
    CollisionInteraction(key_a=0, key_b=1, particle_map=pm)(0, 1)
