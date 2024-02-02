from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.utils import distance
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class CreateEvent:
    source: int
    location: torch.Tensor
    target_class: int

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, CreateEvent) and
            self.source == __o.source and
            torch.all(self.location == __o.location) and
            self.target_class == __o.target_class
        )

    def resolve(self, particle_map):
        i = particle_map.get_inactive_particle()
        particle_map.particles.x[i] = self.location.clone()
        particle_map[i] = self.target_class


class StateCreateInteraction(InteractionRuleBase):
    type = 'state'

    def __init__(self, source, catalyst, target, particle_map):
        super().__init__("state-create-interaction", particle_map=particle_map)
        self.source, self.catalyst, self.target = source, catalyst, target

    def __call__(self, touching, delta, distance):
        interaction_mat = self._compute_interaction_mat(self.source, self.catalyst)
        perm_mat = interaction_mat * touching
        source_is, catalyst = torch.where(perm_mat)
        # print()
        # print('source', source_is, 'catalyst', catalyst)
        # print('source class', self.particle_map.particle_index[source_is])
        # print('catalyst class', self.particle_map.particle_index[catalyst])
        # print('target class', self.target)
        return [
            CreateEvent(
                source=source.item(),
                location=self.particle_map.particles.x[source.item()],
                target_class=self.target,
            ) for source in source_is
        ]

    def resolve(self, event: CreateEvent):
        i = self.particle_map.get_inactive_particle()
        self.particle_map.particles.x[i] = event.location.clone()
        self.particle_map[i] = event.target_class


@dataclass
class TransitionEvent:
    source: int
    target_class: int

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, TransitionEvent) and
            self.source == __o.source and
            self.target_class == __o.target_class
        )

    def resolve(self, particle_map: ParticleMap):
        i = self.source
        particle_map[i] = self.target_class


class StateTransitionInteraction(InteractionRuleBase):
    type = 'state'

    def __init__(self, source, catalyst, target, particle_map):
        super().__init__("state-transition-interaction", particle_map=particle_map)
        self.source, self.catalyst, self.target = source, catalyst, target

    def __call__(self, touching, delta, distance):
        interaction_mat = self._compute_interaction_mat(self.source, self.catalyst)
        perm_mat = interaction_mat * touching
        source_is, catalyst = torch.where(perm_mat)
        # print()
        # print('source', source_is, 'catalyst', catalyst)
        # print('source class', self.particle_map.particle_index[source_is])
        # print('catalyst class', self.particle_map.particle_index[catalyst])
        # print('target class', self.target)
        return [
            TransitionEvent(
                source=source.item(),
                target_class=self.target
            ) for source in source_is
        ]

    def resolve(self, event: TransitionEvent):
        i = event.source
        self.particle_map[i] = event.target_class
