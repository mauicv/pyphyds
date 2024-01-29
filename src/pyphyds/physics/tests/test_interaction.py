from pyphyds.physics.particles import Particles
from pyphyds.physics.interactions.collision_law import CollisionLaw


def test_collision_law_random_particles():
    particles = Particles(3, properties={"size": 1})
    boundary_law = CollisionLaw([particles, particles])
    boundary_law.step()