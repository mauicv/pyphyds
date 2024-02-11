import torch
import numpy as np
from pyphyds.physics.utils import reflect
from pyphyds.physics.laws.base_law import LawBase
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.discrete_particles import DiscreteParticles


class TorusBoundary(LawBase):
    def __init__(self, bounds: tuple, particles: list[Particles]):
        super().__init__('boundary_box')
        if not isinstance(bounds, torch.Tensor):
            bounds = torch.tensor(bounds)
        self.bounds = bounds
        self.particles = particles

    def __call__(self):
        A = self.particles.x[self.particles.x[:, 0] < 0]
        self.particles.x[self.particles.x[:, 0] < 0] += torch.tensor([self.bounds[0], 0])[None, :]
        self.particles.x[self.particles.x[:, 0] > self.bounds[0]] -= torch.tensor([self.bounds[0], 0])[None, :]
        self.particles.x[self.particles.x[:, 1] < 0] += torch.tensor([0, self.bounds[1]])[None, :]
        self.particles.x[self.particles.x[:, 1] > self.bounds[0]] -= torch.tensor([0, self.bounds[1]])[None, :]


class BoxBoundary(LawBase):
    def __init__(self, bounds: tuple, particles: list[Particles]):
        super().__init__('boundary_box')
        if not isinstance(bounds, torch.Tensor):
            bounds = torch.tensor(bounds)
        self.bounds = bounds
        self.particles = particles

    def __call__(self):
        a = torch.where(self.particles.x[:, 0] < 0)[0]
        if a.any():
            self.particles.x[a, 0] = 0
            self.particles.v[a, 0] = -self.particles.v[a, 0]

        a = torch.where(self.particles.x[:, 0] > self.bounds[0])[0]
        if a.any():
            self.particles.x[a, 0] = self.bounds[0].to(self.particles.x.dtype)
            self.particles.v[a, 0] = -self.particles.v[a, 0]

        a = torch.where(self.particles.x[:, 1] < 0)[0]
        if a.any():
            self.particles.x[a, 1] = 0
            self.particles.v[a, 1] = -self.particles.v[a, 1]

        a = torch.where(self.particles.x[:, 1] > self.bounds[1])[0]
        if a.any():
            self.particles.x[a, 1] = self.bounds[1].to(self.particles.x.dtype)
            self.particles.v[a, 1] = -self.particles.v[a, 1]


class DiscreteBoundaryBox(LawBase):
    def __init__(self, bounds: tuple, particles: list[DiscreteParticles]):
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