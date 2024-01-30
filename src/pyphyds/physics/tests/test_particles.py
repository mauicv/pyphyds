import numpy as np
from pyphyds.physics.particles import Particles, ParticleMap

def test_particles():
    p = Particles(1, x_bound=np.array([10, 10]), v_bound=np.array([3, 3]))
    p.x = np.array([5, 5])
    p.v = np.array([1, -1])
    p.step()
    assert (p.x == np.array([6, 4])).all()

def test_particle_map():
    p = Particles(10, x_bound=np.array([10, 10]), v_bound=np.array([3, 3]))
    pm = ParticleMap(p, [0, 1], [0.5, 0.5])
    a = pm.get_class('x', 1)
    pm.set_class('x', 1, np.zeros_like(a))
