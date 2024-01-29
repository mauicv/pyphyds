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
        vect_diffs = p2.x[None, :] - p1.x[:, None]
        D = vect_diffs.norm(dim=-1)
        if self.self_interaction:
            D = D + torch.eye(D.shape[0]) * (self.p1w + self.p2w) + 0.01
        vect_diffs_normed = vect_diffs / D[:, :, None]
        diff = D - (self.p1w + self.p2w)
        diff[diff > 0] = 0
        diff[diff < 0] = 1
        vect_diffs_normed = vect_diffs_normed * diff[:, :, None]

        updates = diff[:, :, None] * vect_diffs
        p1_x_update = updates.sum(dim=1)
        p1_old_x_update = updates.sum(dim=1) 

        p2_x_update = updates.sum(dim=0)
        p2_old_x_update = updates.sum(dim=0)

        p1.x -= p1_x_update * 0.5
        p1.old_x += p1_old_x_update * 0.5

        p2.x += p2_x_update * 0.5
        p2.old_x -= p2_old_x_update * 0.5

        # # print(p1.old_x, p1.x)
        # p1d = p1.old_x - p1.x
        # # print(p1d, vect_diffs_normed, (p1d * vect_diffs_normed).sum(-1))
        # v1 = (p1d * vect_diffs_normed).sum(-1)[:, :, None] * vect_diffs_normed
        # # print('v1.sum(1) =', v1.sum(1))
        # p1.old_x -= v1.sum(1)

        # p2d = p2.old_x - p2.x
        # # print(p1d, vect_diffs_normed, (p1d * vect_diffs_normed).sum(-1))
        # v2 = (p2d * vect_diffs_normed).sum(-1)[:, :, None] * vect_diffs_normed
        # # print('v2.sum(1) =', v2.sum(0))
        # p2.old_x -= v2.sum(0)

        # # assert 1 == 0
        # # print(p1d.shape)
        # pass