
from typing import List
from pyphyds.physics.particles import Particles
from pyphyds.laws.base import UniversalLaw
from pyphyds.interactions.base import InteractionLaw


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


class CellularSimulation:
    def __init__(
            self,
            grid_width: int,
            grid_height: int,
        ):
        pass