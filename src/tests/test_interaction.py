import torch
from pyphyds.particles import Particles
from pyphyds.interactions.collision_law import CollisionLaw


def test_collision_law_random_particles():
    particles = Particles(3, properties={"size": 1})
    boundary_law = CollisionLaw([particles, particles])
    boundary_law.step()


def test_collision_law_random_2_particles():
    particles_1 = Particles(1, properties={"size": 0.5})
    particles_2 = Particles(1, properties={"size": 0.5})

    particles_1.x = torch.tensor([[0.1, 0.1]])
    particles_1.old_x = torch.tensor([[0.1, 0.2]])
    s11 = particles_1.s

    particles_2.x = torch.tensor([[-0.1, -0.1]])
    particles_2.old_x = torch.tensor([[-0.1, -0.2]])
    s21 = particles_2.s

    boundary_law = CollisionLaw([particles_1, particles_2])
    boundary_law.step()

    s12 = particles_1.s
    s22 = particles_2.s
    print(s11, s21, s12, s22)
    print(s11 + s21, s12 + s22)