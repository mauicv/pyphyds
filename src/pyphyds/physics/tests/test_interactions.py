import numpy as np
from pyphyds.physics.particles import Particles, ParticleMap
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction, CollisionInteraction


def test_particles():
    p = Particles(2, x_bound=np.array([10, 10]), v_bound=np.array([3, 3]))
    pm = ParticleMap(p, [0, 1], [0.5, 0.5])
    p.x = np.array([[5, 5], [5, 5]])
    p.v = np.array([[1, -1], [-1, 0]])
    CollisionInteraction(key_a=0, key_b=1, particle_map=pm)(0, 1)
