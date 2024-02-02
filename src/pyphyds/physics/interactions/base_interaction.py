import torch
from pyphyds.physics.particles.particle_map import ParticleMap


class InteractionRuleBase:
    def __init__(self, name: str, particle_map: ParticleMap):
        self.name = name
        self.particle_map = particle_map
        self.particles = particle_map.particles

    def __call__(
            self,
            touching: torch.Tensor,
            delta: torch.Tensor,
            distance: torch.Tensor
        ) -> torch.Tensor:
        raise NotImplementedError()

    def _compute_interaction_mat(self, to_keys: tuple, from_keys: tuple):
        classes = self.particle_map.particle_index
        to_keys_tensor = torch.tensor(to_keys)
        from_keys_tensor = torch.tensor(from_keys)
        to_mask = torch.isin(classes, to_keys_tensor)
        from_mask = torch.isin(classes, from_keys_tensor)
        mask = to_mask[:, None] & from_mask[None, :]
        return mask