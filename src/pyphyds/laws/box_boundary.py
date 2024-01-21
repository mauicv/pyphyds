import torch
from typing import List
from pyphyds.particles import Particles
from pyphyds.laws.base import UniversalLaw


class BoxBoundaryLaw(UniversalLaw):
    def __init__(self,
            particles: Particles,
            name: str,
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float
        ):
        super().__init__(particles, name)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.bounds = (self.x_min, self.x_max, self.y_min, self.y_max)

    def step(self, dt: float):
        # X-axis
        is_oob = self.particles.x[:, 0] < self.x_min
        if is_oob.any():
            d_oob = self.particles.x[is_oob][:, 0] - self.x_min
            d_oob = torch.stack((d_oob, torch.zeros_like(d_oob))).T
            self.particles.x[is_oob] -= 2 * d_oob

            d_oob = self.particles.old_x[is_oob][:, 0] - self.x_min
            d_oob = torch.stack((d_oob, torch.zeros_like(d_oob))).T
            self.particles.old_x[is_oob] -= 2 * d_oob

        is_oob = self.particles.x[:, 0] > self.x_max
        if is_oob.any():
            d_oob = self.particles.x[is_oob][:, 0] - self.x_max
            d_oob = torch.stack((d_oob, torch.zeros_like(d_oob))).T
            self.particles.x[is_oob] -= 2 * d_oob

            d_oob = self.particles.old_x[is_oob][:, 0] - self.x_max
            d_oob = torch.stack((d_oob, torch.zeros_like(d_oob))).T
            self.particles.old_x[is_oob] -= 2 * d_oob

        # Y-axis
        is_oob = self.particles.x[:, 1] < self.y_min
        if is_oob.any():
            d_oob = self.particles.x[is_oob][:, 1] - self.y_min
            d_oob = torch.stack((torch.zeros_like(d_oob), d_oob)).T
            self.particles.x[is_oob] -= 2 * d_oob

            d_oob = self.particles.old_x[is_oob][:, 1] - self.y_min
            d_oob = torch.stack((torch.zeros_like(d_oob), d_oob)).T
            self.particles.old_x[is_oob] -= 2 * d_oob

        is_oob = self.particles.x[:, 1] > self.y_max
        if is_oob.any():
            d_oob = self.particles.x[is_oob][:, 1] - self.y_max
            d_oob = torch.stack((torch.zeros_like(d_oob), d_oob)).T
            self.particles.x[is_oob] -= 2 * d_oob

            d_oob = self.particles.old_x[is_oob][:, 1] - self.y_max
            d_oob = torch.stack((torch.zeros_like(d_oob), d_oob)).T
            self.particles.old_x[is_oob] -= 2 * d_oob