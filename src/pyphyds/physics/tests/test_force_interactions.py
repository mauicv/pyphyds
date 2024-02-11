import torch
from pyphyds.physics.simulation import Simulation
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.interactions.collision_interaction import (
    CollisionInteraction, SeparationInteraction
)
from pyphyds.physics.interactions.force_interaction import ForceInteraction


def test_collision():
    p = Particles(4, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    p.x = torch.tensor([[1., 1.], [9., 1.], [2., 2.], [9., 2.]])
    pm = ParticleMap(
        p, 3, [0.5, 0.5, 0.],
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
        interactions=[collision_interaction],
        laws=[]
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


def test_collision_2():
    p = Particles(2, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    p.x = torch.tensor([[3., 3.], [3., 4.]])
    pm = ParticleMap(
        p, 1, [1.],
        properties={
            1: {'size': 1},
        }
    )
    pm.particle_index = torch.tensor([1, 1])
    collision_interaction = CollisionInteraction(
        keys=[1],
        particle_map=pm
    )
    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[collision_interaction],
        laws=[]
    )
    v = sim.particles.v.clone()
    delta, _ = sim.step()
    assert (v[1] - v[0] == delta[1]).all()
    assert (v[0] - v[1] == delta[0]).all()


# def test_separation():
#     p = Particles(4, x_bound=torch.tensor([10., 10.]), v_bound=3.)
#     p.x = torch.tensor([[1., 1.], [9., 1.], [2., 2.], [9., 2.]])
#     pm = ParticleMap(
#         p, 3, [0.5, 0.5, 0],
#         properties={
#             1: {'size': 1},
#             2: {'size': 0.5},
#             3: {'size': 0.5}
#         }
#     )
#     pm.particle_index = torch.tensor([1, 2, 3, 1])
#     collision_interaction = SeparationInteraction(
#         keys=[1, 2],
#         particle_map=pm
#     )
#     sim = Simulation(
#         particles=p,
#         particle_map=pm,
#         interactions=[collision_interaction],
#         laws=[]
#     )

#     # TODO: Add test for separation interaction
#     sim.step()


def test_force_interaction():
    p = Particles(4, x_bound=torch.tensor([10., 10.]), v_bound=3.)
    p.x = torch.tensor([[1., 1.], [9., 1.], [2., 2.], [9., 2.]])
    pm = ParticleMap(
        p, 3, [0.5, 0.5, 0.],
        properties={
            1: {'size': 1},
            2: {'size': 0.5},
            3: {'size': 0.5}
        }
    )
    pm.particle_index = torch.tensor([1, 0, 3, 1])

    interaction = ForceInteraction(
        keys=[1, 2, 3],
        particle_map=pm,
        force_matrix=torch.tensor([
            [1., 0.5, 0.5],
            [0.5, 1., 0.5],
            [0.5, 0.5, 1.]
        ])
    )

    print(interaction.force_matrix.shape)

    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[interaction],
        laws=[]
    )

    sim.step()