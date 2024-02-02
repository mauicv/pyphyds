import torch
from pyphyds.physics.simulation import Simulation
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.interactions.collision_interaction import (
    CollisionInteraction, SeparationInteraction
)


def test_collision():
    p = Particles(4, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    p.x = torch.tensor([[1., 1.], [9., 1.], [2., 2.], [9., 2.]])
    pm = ParticleMap(
        p, 3, [0.5, 0.5, 0],
        properties={
            1: {'size': 1},
            2: {'size': 0.5},
            3: {'size': 0.5}
        }
    )
    pm.particle_index = torch.tensor([1, 2, 3, 1])
    collision_interaction = CollisionInteraction(
        keys=[1, 2],
        particle_map=pm
    )
    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[collision_interaction]
    )

    interaction_mat = collision_interaction._compute_interaction_mat(
        collision_interaction.keys, 
        collision_interaction.keys
    )

    assert (interaction_mat == torch.tensor([
        [True,True,False,True],
        [True,True,False,True],
        [False,False,False,False],
        [True,True,False,True]
    ])).all()

    sim.step()


def test_separation():
    p = Particles(4, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    p.x = torch.tensor([[1., 1.], [9., 1.], [2., 2.], [9., 2.]])
    pm = ParticleMap(
        p, 3, [0.5, 0.5, 0],
        properties={
            1: {'size': 1},
            2: {'size': 0.5},
            3: {'size': 0.5}
        }
    )
    pm.particle_index = torch.tensor([1, 2, 3, 1])
    collision_interaction = SeparationInteraction(
        keys=[1, 2],
        particle_map=pm
    )
    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[collision_interaction]
    )

    # TODO: Add test for separation interaction
    sim.step()
