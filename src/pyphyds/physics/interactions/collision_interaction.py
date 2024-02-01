from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.utils import distance
import numpy as np


class CollisionInteraction(InteractionRuleBase):
    def __init__(self, keys, particle_map):
        super().__init__("collision-interaction")
        self.keys = keys
        self.particle_map = particle_map
        self.particles = particle_map.particles

    def __call__(self, i, j):
        p_i_class = self.particle_map[i]
        p_j_class = self.particle_map[j]
        if self._is_interaction(p_i_class, p_j_class):
            if (p_i_class == p_j_class) and (i == j):
                # don't allow self collision
                return
            ax = (self.particles.x[i]).clone()
            bx = (self.particles.x[j]).clone()
            sum_of_radius = self.particle_map.get_properties(i)['size'] + \
                self.particle_map.get_properties(j)['size']
            if (distance(ax, bx) - sum_of_radius < 0):
                v_i = self.particles.v[i].clone()
                v_j = self.particles.v[j].clone()
                self.particles.v[i] = v_j
                self.particles.v[j] = v_i
                dab = ax - bx
                self.particles.x[i] = ax + 0.1 * dab
                self.particles.x[j] = bx - 0.1 * dab
    
    def _is_interaction(self, p_i_class, p_j_class):
        if (p_i_class in self.keys) and (p_j_class in self.keys):
            return True


class SeparationInteraction(InteractionRuleBase):
    def __init__(self, keys, particle_map):
        super().__init__("collision-interaction")
        self.keys = keys
        self.particle_map = particle_map
        self.particles = particle_map.particles

    def __call__(self, i, j):
        p_i_class = self.particle_map[i]
        p_j_class = self.particle_map[j]
        if self._is_interaction(p_i_class, p_j_class):
            if (p_i_class == p_j_class) and (i == j):
                # don't allow self collision
                return
            ax = (self.particles.x[i]).clone()
            bx = (self.particles.x[j]).clone()
            sum_of_radius = self.particle_map.get_properties(i)['size'] + \
                self.particle_map.get_properties(j)['size']
            if (distance(ax, bx) - sum_of_radius < 0):
                dab = ax - bx
                self.particles.x[i] = ax + 0.1 * dab
                self.particles.x[j] = bx - 0.1 * dab

    def _is_interaction(self, p_i_class, p_j_class):
        if (p_i_class in self.keys) and (p_j_class in self.keys):
            return True


class DiscreteCollisionInteraction(InteractionRuleBase):
    def __init__(self, key_a, key_b, particle_map):
        super().__init__("collision-interaction")
        self.key_a = key_a
        self.key_b = key_b
        self.particle_map = particle_map
        self.particles = particle_map.particles

    def __call__(self, i, j):
        p_i_class = self.particle_map[i]
        p_j_class = self.particle_map[j]
        if (p_i_class == self.key_a and p_j_class == self.key_b) \
                or (p_i_class == self.key_b and p_j_class == self.key_a):
            ax = np.copy(self.particles.x[i])
            bx = np.copy(self.particles.x[j])
            if (ax == bx).all():
                v_i = np.copy(self.particles.v[i])
                v_j = np.copy(self.particles.v[j])
                self.particles.v[i] = v_j
                self.particles.v[j] = v_i
