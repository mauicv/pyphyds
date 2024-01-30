import numpy as np
from pyphyds.physics.particles import Particles

def test_particles():
    p = Particles(1, x_bound=np.array([10, 10]), v_bound=np.array([3, 3]))
    p.x = np.array([5, 5])
    p.v = np.array([1, -1])
    p.step()
    assert (p.x == np.array([6, 4])).all()