from pyphyds.physics.interactions.base_interaction import InteractionRuleBase
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.utils import equi_points
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class DeathEvent:
    source: int

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, CreateEvent) and
            self.source == __o.source
        )

    def resolve(self, particle_map):
        particle_map[self.source] = 0


@dataclass
class CreateEvent:
    location: torch.Tensor
    target_classes: list[int]

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, CreateEvent) and
            self.source == __o.source and
            torch.all(self.location == __o.location) and
            self.target_classes == __o.target_classes
        )

    def resolve(self, particle_map: ParticleMap):
        points = equi_points(len(self.target_classes))
        inactive_particles = particle_map.get_inactive_particles()
        if len(inactive_particles) >= 2:
            for target_class, vector, i in zip(self.target_classes, points, inactive_particles):
                r = 2 * particle_map.properties[target_class]['size']
                particle_map.particles.x[i] = self.location.clone() + vector * r
                particle_map.particles.v[i] = vector
                particle_map[i] = target_class


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
        return [
            TransitionEvent(
                source=source.item(),
                target_class=self.target
            ) for source in source_is
        ]

    def resolve(self, event: TransitionEvent):
        i = event.source
        self.particle_map[i] = event.target_class


class SpontaneousTransitionInteraction(InteractionRuleBase):
    type = 'state'

    def __init__(
            self,
            source,
            targets,
            particle_map,
            event_probability
        ):
        super().__init__("spontaneous-transition-interaction", particle_map=particle_map)
        self.source, self.targets = source, targets
        self.probability = event_probability

    def __call__(self, touching, delta, distance):
        source_num = self.particle_map.particle_counts[self.source]
        p = torch.rand(source_num)
        p_bool = torch.where(p < self.probability)[0]
        if len(p_bool) == 0:
            return []
        source_inds = self.particle_map.get_class_instances(self.source)
        events = []
        sources = source_inds[p_bool]
        for i in sources:
            events.extend([
                DeathEvent(
                    source=i.item(),
                ),
                CreateEvent(
                    location=self.particle_map.particles.x[i],
                    target_classes=self.targets
                )
            ])
        return events
