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
        if len(self.force_interactions) > 0:
            delta_v_acc = torch.zeros_like(self.particles.v)
            for interaction in self.force_interactions:
                delta_v = interaction(touching, delta, distance)
                delta_v_acc += delta_v

        if len(self.position_interactions) > 0:
            delta_x_acc = torch.zeros_like(self.particles.x)
            for interaction in self.position_interactions:
                delta_x = interaction(touching, delta, distance)
                delta_x_acc += delta_x

        state_events = []
        for interaction in self.state_interactions:
            state_events.extend(
                interaction(touching, delta, distance)
            )

        if len(self.force_interactions) > 0:
            self.particles.v -= delta_v_acc

        if len(self.position_interactions) > 0:
            self.particles.x += delta_x_acc

        for event in state_events:
            event.resolve(self.particle_map)

        for law in self.laws:
            law()

        self.particles.step()
        return delta_v_acc, state_events

    def _compute_proximities(self):
        delta = self.particles.x[None, :] - self.particles.x[:, None]
        distance = (delta**2).sum(-1).sqrt()
        sizes = self.particle_map.get_prop_vect('size')
        size_mat = sizes[None, :] + sizes[:, None]
        D = distance - size_mat + 2.001*torch.eye(self.particles.number)*sizes
        touching = D < 0
        return distance, delta, touching
