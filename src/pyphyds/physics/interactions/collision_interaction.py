from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
import numpy as np


class CollisionInteraction(InteractionRuleBase):
    def __init__(self, particles_a, particles_b):
        super().__init__("CollisionInteraction")
        self.particles_a = particles_a
        self.particles_b = particles_b
        assert self.particles_a != self.particles_b

    def __call__(self):
        for i, ax in enumerate(self.particles_a.x):
            for j, bx in enumerate(self.particles_b.x):
                if (ax == bx).all():
                    v_i = np.copy(self.particles_a.v[i])
                    v_j = np.copy(self.particles_b.v[j])
                    self.particles_a.v[i] = v_j
                    self.particles_b.v[j] = v_i


class SelfCollisionInteraction(InteractionRuleBase):
    def __init__(self, particles):
        super().__init__("SelfCollisionInteraction")
        self.particles = particles

    def __call__(self):
        for i, ax in enumerate(self.particles.x):
            for j in range(i+1, self.particles.number):
                if (ax == self.particles.x[j]).all():
                    v_i = np.copy(self.particles.v[i])
                    v_j = np.copy(self.particles.v[j])
                    self.particles.v[i] = v_j
                    self.particles.v[j] = v_i