import torch
from typing import List

from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.particles.discrete_particles import DiscreteParticles
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap


class Simulation:
    def __init__(
            self,
            particles: Particles,
            particle_map: ParticleMap,
            interactions: List[InteractionRuleBase] = None,
            laws: List[InteractionRuleBase] = None,
        ):
        self.particles = particles
        self.particle_map = particle_map
        self.laws = laws
        self.force_interactions = [interaction for interaction in interactions if interaction.type == 'force']
        self.position_interactions = [interaction for interaction in interactions if interaction.type == 'position']
        self.state_interactions = [interaction for interaction in interactions if interaction.type == 'state']

    def step(self):
        distance, delta, touching = self._compute_proximities()
        state_events = []
        for interaction in self.state_interactions:
            state_events.extend(
                interaction(touching, delta, distance)
            )

        for event in state_events:
            event.resolve(self.particle_map)
            """
            # EVENT LOG:
            # TransitionEvent(source=1, target_class=3)
            # TransitionEvent(source=4, target_class=0)
            #
            # TransitionEvent(source=1, target_class=2)
            # CreateEvent(source=1, location=tensor([132.7887, 244.3631]), target_class=1)
            #
            # TransitionEvent(source=4, target_class=3)
            # TransitionEvent(source=1, target_class=0)
            #
            # The First two sets of two are correct the last on is also correct
            # but occurs straight after the second set of two. This is because
            # the particle materializes and then reacts straight away. This is
            # not the desired behavior.
            # """

        delta_v_acc = torch.zeros_like(self.particles.v)
        if len(self.force_interactions) > 0:
            for interaction in self.force_interactions:
                delta_v = interaction(touching, delta, distance)
                delta_v_acc += delta_v

        delta_x_acc = torch.zeros_like(self.particles.x)
        if len(self.position_interactions) > 0:
            for interaction in self.position_interactions:
                delta_x = interaction(touching, delta, distance)
                delta_x_acc += delta_x

        if len(self.force_interactions) > 0:
            self.particles.x -= delta_v_acc

        # if len(self.force_interactions) > 0:
        #     self.particles.v -= delta_v_acc

        # if len(self.position_interactions) > 0:
        #     self.particles.x += delta_x_acc

        for law in self.laws:
            law()

        actvice_inds = self.particle_map.get_active_particle_indices()
        self.particles.step(actvice_inds)
        return delta_v_acc, state_events

    def _compute_proximities(self):
        delta = self.particles.x[None, :] - self.particles.x[:, None]
        distance = (delta**2).sum(-1).sqrt()
        sizes = self.particle_map.get_prop_vect('size')
        size_mat = sizes[None, :] + sizes[:, None]
        D = distance - size_mat + 2.001*torch.eye(self.particles.number)*sizes
        touching = D < 0
        return distance, delta, touching
