import torch
from typing import List

from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.particles import Particles


class Simulation:
    def __init__(
            self,
            particles: List[Particles],
            interactions: List[InteractionRuleBase] = None,
            laws: List[InteractionRuleBase] = None,
        ):
        self.particles = particles
        self.laws = laws
        self.interaction_rules = interactions

    def step(self):
        for p in self.particles:
            p.step()

        for law in self.laws:
            law()