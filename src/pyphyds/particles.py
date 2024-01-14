from typing import Dict
import torch


class Particles:
    def __init__(
            self,
            num: int,
            properties: Dict=None,
            id: str='none'
        ):
        self.x = torch.randn(num, 2)
        self.old_x = torch.randn(num, 2)
        self.properties = properties
        self.num = num
        self.id = id

    def dx(
            self,
            dt: torch.tensor
        ):
        dv = self.x - self.old_x
        self.old_x += dv * dt
        self.x += dv * dt

    def v(self):
        return self.x - self.old_x

    def __repr__(self):
        return f"Particles(x.shape={self.x.shape}, num={self.num})"