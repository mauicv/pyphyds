from typing import List, Any, Dict, Optional, Union
import numpy as np
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.discrete_particles import DiscreteParticles


class ParticleMap:
    def __init__(
            self,
            particles: Union[Particles, DiscreteParticles],
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

    def set_properties(self, particle_id: int, properties: Dict):
        self.properties[self.particle_index[particle_id]] = {
            **self.properties[self.particle_index[particle_id]],
            **properties
        }

    def __getitem__(self, key):
        return self.particle_index[key]

    def __setitem__(self, key, value):
        self.particle_index[key] = value