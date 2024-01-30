from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
import numpy as np


class CollisionInteraction(InteractionRuleBase):
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
