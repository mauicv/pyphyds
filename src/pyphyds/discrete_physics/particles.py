from typing import Dict, Optional, List, Any
import numpy as np


class Particles:
    def __init__(
            self,
            number: int,
            x_bound: np.ndarray,
            v_bound: np.ndarray,
            attributes: Optional[Dict]=None,
        ) -> None:
        self.attributes = attributes
        self.number = number
        self.x_bound = np.array(x_bound)
        self.x = np.random.randint(np.array([0,0]), self.x_bound, (number, 2))
        self.v = np.random.randint(-v_bound, v_bound + 1, (number, 2))

    def step(self):
        self.x += self.v


class ParticleMap:
    def __init__(
            self,
            particles: Particles,
            classes: List[Any],
            probs: List[float]
        ):
        self.particles = particles
        self.classes = classes
        self.particle_index = np.random.choice(
            classes,
            size=self.particles.number,
            p=probs,
            replace=True
        )

    def get_class(self, attribute: str, key: int):
        return getattr(self.particles, attribute)[self.particle_index == key]

    def set_class(self, attribute: str, key: int, value: Any):
        getattr(self.particles, attribute)[self.particle_index == key] = value

    def __getitem__(self, key):
        return self.particle_index[key]

    def __setitem__(self, key, value):
        self.particle_index[key] = value