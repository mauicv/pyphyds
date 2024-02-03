from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.utils import distance
import numpy as np
import torch


class CollisionInteraction(InteractionRuleBase):
    type = 'force'

    def __init__(self, keys, particle_map):
        super().__init__("collision-interaction", particle_map=particle_map)
        self.keys = keys

    def __call__(self, touching, delta, distance):
        interaction_mat = self._compute_interaction_mat(self.keys, self.keys)
        perm_mat = interaction_mat * touching
        touching_bool = touching.sum(-1)
        stop_v = touching_bool[:, None] * self.particle_map.particles.v
        delta_v = (
            perm_mat
            .to(torch.double)
            @ self.particle_map.particles.v
            .to(torch.double)
        )
        return stop_v - delta_v


class SeparationInteraction(InteractionRuleBase):
    type = 'force'

    def __init__(self, keys, particle_map):
        super().__init__("collision-interaction", particle_map=particle_map)
        self.keys = keys
        self.particle_map = particle_map
        self.particles = particle_map.particles

    def __call__(self, touching, delta, distance):
        interaction_mat = self._compute_interaction_mat(self.keys, self.keys)
        perm_mat = (interaction_mat * touching)
        n_delta = torch.zeros_like(delta)
        n_delta[perm_mat] = delta[perm_mat]
        return n_delta
        


# class DiscreteCollisionInteraction(InteractionRuleBase):
#     def __init__(self, key_a, key_b, particle_map):
#         super().__init__("collision-interaction")
#         self.key_a = key_a
#         self.key_b = key_b
#         self.particle_map = particle_map
#         self.particles = particle_map.particles

#     def __call__(self, i, j):
#         p_i_class = self.particle_map[i]
#         p_j_class = self.particle_map[j]
#         if (p_i_class == self.key_a and p_j_class == self.key_b) \
#                 or (p_i_class == self.key_b and p_j_class == self.key_a):
#             ax = np.copy(self.particles.x[i])
#             bx = np.copy(self.particles.x[j])
#             if (ax == bx).all():
#                 v_i = np.copy(self.particles.v[i])
#                 v_j = np.copy(self.particles.v[j])
#                 self.particles.v[i] = v_j
#                 self.particles.v[j] = v_i
