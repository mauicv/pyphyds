from typing import List
from pyphyds.particles import Particles
import torch


class UniversalLaw:
    def __init__(self,
            particles: Particles,
            name: str
        ):
        self.name = name
        self.particles = particles

    def step(self, dt: float):
        raise NotImplementedError()
