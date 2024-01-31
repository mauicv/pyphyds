import torch
from pyphyds.physics.utils import reflect
from pyphyds.physics.laws.base_law import LawBase
from pyphyds.physics.particles import Particles


class BoundaryBox(LawBase):
    def __init__(self, bounds: tuple, particles: list[Particles]):
        super().__init__('boundary_box')
        if not isinstance(bounds, torch.Tensor):
            bounds = torch.tensor(bounds)
        self.bounds = bounds
        self.particles = particles

    def __call__(self):
        for particle in self.particles:
            x_v = particle.v.clone()
            for i, (p, v) in enumerate(zip(particle.x, particle.v)):
                if p[0] < 0 or p[0] > self.bounds[0]:
                    x_v[i] = reflect(v, torch.tensor([1., 0.]))
                if p[1] < 0 or p[1] > self.bounds[1]:
                    x_v[i] = reflect(v, torch.tensor([0., 1.]))
            particle.v = x_v
