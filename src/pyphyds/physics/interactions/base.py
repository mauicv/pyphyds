from typing import List
from pyphyds.physics.particles import Particles
import torch


class InteractionLaw:
    def __init__(
            self,
            particles: List[Particles],
            name: str
        ):
        self.name = name
        self.particles = particles

    def step(self):
        raise NotImplementedError()