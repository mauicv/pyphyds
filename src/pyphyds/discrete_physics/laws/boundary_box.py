import numpy as np
from pyphyds.discrete_physics.laws.base_law import LawBase
from pyphyds.discrete_physics.particles import Particles


class BoundaryBox(LawBase):
    def __init__(self, bounds: tuple, particles: list[Particles]):
        super().__init__('boundary_box')
        self.bounds = np.array(bounds)
        self.particles = particles

    def __call__(self):
        for particle in self.particles:
            x = np.copy(particle.x)
            x_v = np.copy(particle.v)
            for i, (p, v) in enumerate(zip(particle.x, particle.v)):
                if p[0] < 0 or p[0] > self.bounds[0]:
                    x_v[i] = self.reflect_x(v)
                if p[1] < 0 or p[1] > self.bounds[1]:
                    x_v[i] = self.reflect_y(v)
            particle.v = x_v

    def reflect_x(self, v):
        return np.array([-v[0], v[1]])

    def reflect_y(self, v):
        return np.array([v[0], -v[1]])