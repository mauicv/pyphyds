import numpy as np
from pyphyds.physics.particles import Particles
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction, SelfCollisionInteraction


def test_particles():
    p = Particles(2, x_bound=np.array([10, 10]), v_bound=np.array([3, 3]))
    p.x = np.array([[5, 5], [5, 5]])
    p.v = np.array([[1, -1], [-1, 0]])
    SelfCollisionInteraction(p)()
    print(p.v)
    # assert (p.v == np.array([[-1, 1], [1, -1]])).all()  