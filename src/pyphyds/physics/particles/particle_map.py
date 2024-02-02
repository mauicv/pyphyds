from typing import List, Any, Dict, Optional, Union
import torch
from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.discrete_particles import DiscreteParticles


class ParticleMap:
    def __init__(
            self,
            particles: Union[Particles, DiscreteParticles],
            classes: int,
            probs: Union[torch.Tensor, List[float]],
            properties: Optional[Dict]=None
        ):
        self.particles = particles
        assert len(probs) == classes
        self.classes = classes + 1 # 0 is reserved for inactive particles
        if isinstance(probs, list):
            probs = torch.tensor(probs)

        self.particle_index = torch.multinomial(
            probs,
            self.particles.number,
            replacement=True
        )
        if 0 in properties:
            raise ValueError('0 is a reserved class index')

        self.properties = {
            **properties,
            0: {'is_active': False, 'size': 0}
        }

    # def get_class(self, attribute: str, key: int):
    #     return getattr(self.particles, attribute)[self.particle_index == key]

    # def set_class(self, attribute: str, key: int, value: Any):
    #     getattr(self.particles, attribute)[self.particle_index == key] = value

    def get_properties(self, particle_id: int):
        return self.properties[self.particle_index[particle_id]]

    def set_properties(self, particle_id: int, properties: Dict):
        self.properties[self.particle_index[particle_id]] = {
            **self.properties[self.particle_index[particle_id]],
            **properties
        }

    def get_prop_vect(self, property):
        return torch.tensor([self.properties[i.item()][property] for i in self.particle_index])

    def __getitem__(self, key):
        return self.particle_index[key]

    def __setitem__(self, key, value):
        self.particle_index[key] = value

    def get_inactive_particle(self):
        return torch.where(self.particle_index == 0)[0]

    def get_class_instances(self, key: int):
        return torch.where(self.particle_index == key)