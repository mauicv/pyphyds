from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.utils import distance
import numpy as np


class LocalInteraction(InteractionRuleBase):
    def __init__(self, key_a, key_b, particle_map):
        super().__init__("weird-interaction")
        self.key_a = key_a
        self.key_b = key_b
        self.particle_map = particle_map
        self.particles = particle_map.particles

    def __call__(self, i, j):
        p_i_class = self.particle_map[i]
        p_j_class = self.particle_map[j]
        prop_i = self.particle_map.get_properties(i)
        prop_j = self.particle_map.get_properties(j)
        if self._check_ab(p_i_class, p_j_class):
            is_collision, ax, ab = self._check_collision(i, prop_i, j, prop_j)
            if is_collision:
                self._interaction(i, p_i_class, ax, j, p_j_class, ab)

    def _check_collision(self, i, prop_i, j, prop_j):
        ax = (self.particles.x[i]).clone()
        bx = (self.particles.x[j]).clone()
        sum_of_radius = prop_i['size'] + prop_j['size']
        if distance(ax, bx) - sum_of_radius < 0:
            return True, ax, bx
        return False, None, None

    def _check_ab(self, p_i_class, p_j_class):
        return (p_i_class, p_j_class) == (self.key_a, self.key_b) or \
            (p_i_class, p_j_class) == (self.key_b, self.key_a)

    def _interaction(self, i, p_i_class, ax, j, p_j_class, ab):
        compute_map = {
            (self.key_a, self.key_b): self._ab_interaction,
            (self.key_b, self.key_a): self._ba_interaction,
        }
        compute_map[(p_i_class, p_j_class)](i, ax, j, ab)

    def _ab_interaction(self, i, ax, j, ab):
        raise NotImplementedError

    def _ba_interaction(self, i, ax, j, ab):
        raise NotImplementedError
        

class ABtoCDInteraction(LocalInteraction):
    def __init__(self, key_a, key_b, key_c, key_d, particle_map):
        super().__init__(key_a, key_b, particle_map)
        self.key_c = key_c
        self.key_d = key_d

    def _ab_interaction(self, i, ax, j, ab):
        self.particle_map[i] = self.key_c
        self.particle_map[j] = self.key_d

    def _ba_interaction(self, i, ax, j, ab):
        self.particle_map[i] = self.key_c
        self.particle_map[j] = self.key_d
        pass


# class CDtoABInteraction(LocalInteraction):
#     def __init__(self, key_a, key_b, particle_map):
#         super().__init__(key_a, key_b, particle_map)

#     def _ab_interaction(self, i, ax, j, ab):
#         self.particle_map[j] = self.key_b
#         self.particle_map.set_properties(j, {'active': True})

#     def _ba_interaction(self, i, ax, j, ab):
#         self.particle_map[j] = self.key_b
#         self.particle_map.set_properties(j, {'active': True})