
from typing import List
from pyphyds.physics.particles import Particles
from pyphyds.physics.laws.base import UniversalLaw
from pyphyds.physics.interactions.base import InteractionLaw


class PhysicsSimulation:
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
            law.step()

        for interaction in self.interactions:
            interaction.step()
