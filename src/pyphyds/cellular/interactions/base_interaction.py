import torch


class InteractionRuleBase:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()