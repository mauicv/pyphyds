import torch
from pyphyds.physics.simulation import Simulation
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction, SeparationInteraction
from pyphyds.physics.interactions.local_interaction import (
    StateTransitionInteraction,
    CreateEvent,
    TransitionEvent,
    SpontaneousTransitionInteraction
)


# def test_state_create():
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
#     pm.particle_index = torch.tensor([1, 2, 0, 1])
#     interaction = StateCreateInteraction(
#         source=1,
#         catalyst=2,
#         target=3,
#         particle_map=pm
#     )

#     interaction_mat = interaction._compute_interaction_mat(
#         interaction.source,
#         interaction.catalyst
#     )

#     t = torch.tensor([
#         [False,  True, False, False],
#         [False, False, False, False],
#         [False, False, False, False],
#         [False,  True, False, False]
#     ])

#     assert (t == interaction_mat).all()

#     sim = Simulation(
#         particles=p,
#         particle_map=pm,
#         interactions=[interaction]
#     )

#     distance, delta, touching = sim._compute_proximities()
#     events = interaction(touching, delta, distance)
#     assert events == [
#         CreateEvent(source=3, target_class=3, location=p.x[3])
#     ]
#     for event in events:
#         interaction.resolve(event)

#     assert (pm.particle_index == torch.tensor([1, 2, 3, 1])).all()


def test_state_transition():
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
    pm.particle_index = torch.tensor([1, 2, 0, 1])
    interaction = StateTransitionInteraction(
        source=1,
        catalyst=2,
        target=3,
        particle_map=pm
    )

    interaction_mat = interaction._compute_interaction_mat(
        interaction.source,
        interaction.catalyst
    )

    t = torch.tensor([
        [False,  True, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False,  True, False, False]
    ])

    assert (t == interaction_mat).all()

    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[interaction]
    )

    distance, delta, touching = sim._compute_proximities()
    events = interaction(touching, delta, distance)
    assert events == [
        TransitionEvent(source=3, target_class=3)
    ]
    for event in events:
        interaction.resolve(event)

    assert (pm.particle_index == torch.tensor([1, 2, 0, 3])).all()


def test_spontaneous_transition():
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
    pm.particle_index = torch.tensor([1, 2, 0, 1])
    interaction = SpontaneousTransitionInteraction(
        source=1,
        targets=[2, 3],
        particle_map=pm,
        event_probability=0.5
    )

    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[interaction]
    )

    distance, delta, touching = sim._compute_proximities()
    events = interaction(touching, delta, distance)

    for event in events:
        event.resolve(pm)

    # assert (pm.particle_index == torch.tensor([1, 2, 0, 3])).all()