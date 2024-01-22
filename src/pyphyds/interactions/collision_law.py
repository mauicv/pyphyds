from typing import List
from pyphyds.particles import Particles
from pyphyds.interactions.base import InteractionLaw
import torch


class CollisionLaw(InteractionLaw):
    def __init__(
            self,
            particles: List[Particles],
        ):
        assert len(particles) == 2, "Collision interaction law requires exactly two sets of particles"
        for particle in particles:
            assert 'size' in particle.properties
        self.self_interaction = True if particles[0] is particles[1] else False
        self.p1w = particles[0].properties['size']
        self.p2w = particles[1].properties['size']
        super().__init__(particles, "Collision")

    def step(self):
        p1, p2 = self.particles
        vect_diffs = p1.x[None, :] - p2.x[:, None]
        D = vect_diffs.norm(dim=-1)
        if self.self_interaction:
            D = D + torch.eye(D.shape[0]) * (self.p1w + self.p2w) + 0.01
        diff = D - (self.p1w + self.p2w)
        diff[diff > 0] = 0
        diff[diff < 0] = 1/2
        updates = diff[:, :, None] * vect_diffs
        p1.x -= updates.sum(dim=0)
        p2.x += updates.sum(dim=1)
