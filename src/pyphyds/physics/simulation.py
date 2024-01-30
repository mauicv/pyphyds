import torch
from typing import List

from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.particles import Particles, ParticleMap


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
        self.interactions = interactions

    def step(self):
        self.particles.step()
        for p_i in range(self.particles.number):
            for p_j in range(p_i, self.particles.number):
                for interaction in self.interactions:
                    interaction(p_i, p_j)

        for law in self.laws:
            law()
