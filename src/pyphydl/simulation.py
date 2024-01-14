from typing import List
from pyphydl.particles import Particles
from pyphydl.laws import UniversalLaw
from pyphydl.interactions import InteractionLaw


class Simulation:
    def __init__(
            self,
            particles: List[Particles],
            laws: List[UniversalLaw],
            interactions: List[InteractionLaw]
        ):
        self.particles = particles
        self.laws = laws
        self.interactions = interactions
    
    def step(self, dt):
        for particle in self.particles:
            particle.dx(dt)

        for law in self.laws:
            law.step(dt)

        for interaction in self.interactions:
            interaction.step(dt)
