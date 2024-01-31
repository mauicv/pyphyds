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
