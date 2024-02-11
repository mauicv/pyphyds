from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.utils import rotate
import numpy as np
import torch


class ForceInteraction(InteractionRuleBase):
    type = 'force'

    def __init__(
            self,
            keys,
            particle_map: ParticleMap,
            force_matrix: torch.Tensor=None,
        ):
        super().__init__("force-interaction", particle_map=particle_map)
        self.keys = keys
        if force_matrix is not None:
            # Need to add zeros for the non-interacting particles
            n, m = force_matrix.shape
            a = torch.zeros(n + 1, m + 1)
            a[1:, 1:] = force_matrix
            self.force_matrix = a
        else:
            self.force_matrix = torch.randn((len(keys) + 1, len(keys) + 1))
            self.force_matrix[0, :] = 0
            self.force_matrix[:, 0] = 0

    def __call__(self, touching, delta, distance):
        t = self.particle_map.particle_index
        F = self.force_matrix[t][:, t]
        n_d = (delta)/(distance[:, :, None] + 1e-6)
        F[(distance<10)] = -1
        F[(distance>100)] = 1e-6
        v = (F[:, :, None] * n_d).sum(0)
        return v


class AngularForceInteraction(InteractionRuleBase):
    type = 'force'

    def __init__(
            self,
            keys,
            particle_map: ParticleMap,
            force_matrix: torch.Tensor=None,
        ):
        super().__init__("force-interaction", particle_map=particle_map)
        self.keys = keys
        if force_matrix is not None:
            # Need to add zeros for the non-interacting particles
            n, m = force_matrix.shape
            a = torch.zeros(n + 1, m + 1)
            a[1:, 1:] = force_matrix
            self.force_matrix = a
        else:
            self.force_matrix = torch.randn((len(keys) + 1, len(keys) + 1))
            self.force_matrix[0, :] = 0
            self.force_matrix[:, 0] = 0

    def __call__(self, touching, delta, distance):
        t = self.particle_map.particle_index
        F = self.force_matrix[t][:, t]
        n_d = (delta)/(distance[:, :, None] + 1e-6)
        # F[(distance<10)] = -1
        F[(distance>100)] = 1e-6
        v = (F[:, :, None] * n_d).sum(0)
        return rotate(v)

