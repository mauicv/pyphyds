from typing import Dict, Optional
import torch


class Particles:
    def __init__(
            self,
            num: int,
            properties: Optional[Dict]=None,
            id: str='none'
        ):
        self.x = torch.randn(num, 2)
        self.old_x = self.x + torch.randn(num, 2) * 0.0001
        self.properties = properties
        if properties is None:
            self.properties = {}
        self.num = num
        self.id = id

    def dx(
            self,
            dt: torch.tensor
        ):
        dv = self.x - self.old_x
        self.old_x += dv * dt
        self.x += dv * dt

    @property
    def v(self):
        return self.x - self.old_x

    @property
    def s(self):
        return torch.linalg.norm(self.v, dim=-1)

    def __repr__(self):
        return f"Particles(x.shape={self.x.shape}, num={self.num})"