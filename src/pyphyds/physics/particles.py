from random import randint
import numpy as np


class Particles:
    def __init__(self, number, x_bound, v_bound) -> None:
        self.number = number
        self.x_bound = np.array(x_bound)
        self.x = np.random.randint(np.array([0,0]), self.x_bound, (number, 2))
        self.v = np.random.randint(-v_bound, v_bound, (number, 2))

    def step(self):
        self.x += self.v