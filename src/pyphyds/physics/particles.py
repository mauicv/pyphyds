from typing import Dict, Optional, List, Any
import numpy as np
import torch


class Particles:
    def __init__(
            self,
            number: int,
            x_bound: torch.Tensor,
            v_bound: float,
            attributes: Optional[Dict]=None,
        ) -> None:
        self.attributes = attributes
        self.number = number
        if not isinstance(x_bound, torch.Tensor):
            x_bound = torch.tensor(x_bound)
        self.x_bound = x_bound
        self.x = torch.rand((number, 2)) * self.x_bound
        self.v = torch.randn((number, 2)) * v_bound

    def step(self):
        self.x += self.v


class ParticleMap:
    def __init__(
            self,
            particles: Particles,
            classes: List[Any],
            probs: List[float],
            properties: Optional[Dict]=None
        ):
        self.particles = particles
        self.classes = classes
        self.particle_index = np.random.choice(
            classes,
            size=self.particles.number,
            p=probs,
            replace=True
        )
        self.properties = properties

    def get_class(self, attribute: str, key: int):
        return getattr(self.particles, attribute)[self.particle_index == key]

    def set_class(self, attribute: str, key: int, value: Any):
        getattr(self.particles, attribute)[self.particle_index == key] = value

    def get_properties(self, particle_id: int):
        return self.properties[self.particle_index[particle_id]]

    def __getitem__(self, key):
        return self.particle_index[key]

    def __setitem__(self, key, value):
        self.particle_index[key] = value