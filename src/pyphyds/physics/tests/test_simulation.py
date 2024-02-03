import torch
from pyphyds.physics.simulation import Simulation
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.interactions.collision_interaction import (
    CollisionInteraction, SeparationInteraction
)
from pyphyds.physics.interactions.local_interaction import (
    StateCreateInteraction,
    StateTransitionInteraction,
    CreateEvent,
    TransitionEvent
)

# TODO: add fixtures

def test_deltas():
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

    # separation_interaction = SeparationInteraction(
    #     keys=[1, 2],
    #     particle_map=pm
    # )

    state_create_interaction_1 = StateTransitionInteraction(
        source=1,
        catalyst=2,
        target=3,
        particle_map=pm
    )

    state_create_interaction_2 = StateTransitionInteraction(
        source=2,
        catalyst=1,
        target=0,
        particle_map=pm
    )
    
    state_create_interaction_3 = StateCreateInteraction(
        source=3,
        catalyst=1,
        target=2,
        particle_map=pm
    )

    state_transition_interaction = StateTransitionInteraction(
        source=3,
        catalyst=1,
        target=1,
        particle_map=pm
    )

    sim = Simulation(
        particles=p,
        particle_map=pm,
        interactions=[
            collision_interaction,
            # separation_interaction,
            state_create_interaction_1,
            state_create_interaction_2,
            state_create_interaction_3,
            state_transition_interaction
        ],
        laws=[]
    )

    delta, distance, touching = sim._compute_proximities()

    assert (touching == torch.tensor([
        [False, False,  True, False],
        [False, False, False,  True],
        [ True, False, False, False],
        [False,  True, False, False]
    ])).all()

    state_events = []
    for interaction in sim.state_interactions:
        state_events.extend(
            interaction(touching, delta, distance)
        )

    assert state_events[0].source == 3
    assert state_events[0].target_class == 3
    assert isinstance(state_events[0], TransitionEvent)

    assert state_events[1].source == 1
    assert state_events[1].target_class == 0
    assert isinstance(state_events[1], TransitionEvent)

    assert state_events[2].source == 2
    assert state_events[2].target_class == 2
    assert isinstance(state_events[2], CreateEvent)

    assert state_events[3].source == 2
    assert state_events[3].target_class == 1
    assert isinstance(state_events[3], TransitionEvent)

    sim.step()