from typing import Dict, Optional, List, Any
import numpy as np


class DiscreteParticles:
    def __init__(
            self,
            number: int,
            x_bound: np.ndarray,
            v_bound: np.ndarray,
            attributes: Optional[Dict]=None,
        ) -> None:
        self.attributes = attributes
        self.number = number
        self.x_bound = np.array(x_bound)
        self.x = np.random.randint(np.array([0,0]), self.x_bound, (number, 2))
        self.v = np.random.randint(-v_bound, v_bound + 1, (number, 2))

    def step(self):
        self.x += self.v