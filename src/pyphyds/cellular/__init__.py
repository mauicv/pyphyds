import torch
from typing import List

from pyphyds.cellular.interactions.base_interaction import InteractionRuleBase


class CellularSimulation:
    def __init__(
            self,
            width: int,
            height: int,
            num_states: int = 1,
            num_histories: int = 1,
            interaction_rules: List[InteractionRuleBase] = None
        ):
        self.num_states = num_states
        self.num_histories = num_histories
        self.width = width
        self.height = height
        self.states = torch.randint(0, num_states, (num_histories, width, height))
        self.interaction_rules = interaction_rules

    def step(self):
        for rule in self.interaction_rules:
            self.states = rule(self.states)
