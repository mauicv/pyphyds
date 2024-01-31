from typing import List
import numpy as np
from pyphyds.physics.laws.base_law import LawBase
from pyphyds.physics.particles.discrete_particles import DiscreteParticles


class RandomnessLaw(LawBase):
    """Adds randomness to the particles' velocities.

    This is intended for discrete systems which can be very deterministic.
    """
    def __init__(
            self,
            particles: List[DiscreteParticles],
            p: float = 0.01,
            v_bound: np.ndarray=np.array([2, 2])
        ):
        super().__init__('randomness-law')
        self.particles = particles
        self.p = p
        self.v_bound = v_bound

    def __call__(self):
        for particles in self.particles:
            to_perturb = (
                np.random
                .binomial(1, self.p, particles.number)
                .astype(bool)
            )
            new_v = np.random.randint(
                -self.v_bound,
                self.v_bound + 1,
                (to_perturb.sum(), 2)
            )
            particles.v[to_perturb] = new_v
