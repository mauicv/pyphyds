from pyphyds.physics.particles import Particles
from pyphyds.physics.laws.boundary_box import BoundaryBox
from pyphyds.physics.interactions.collision_interaction import SelfCollisionInteraction
from pyphyds.physics import Simulation
import cv2
import numpy as np


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 200
BOUNDS = np.array((SIZE, SIZE))
NUM_PARTICLES = 100

particles = Particles(NUM_PARTICLES, BOUNDS, 2)
simulation = Simulation(
    particles=[particles],
    laws=[BoundaryBox(BOUNDS, [particles])],
    interactions=[SelfCollisionInteraction(particles)]
)


def draw(particles):
    img = np.zeros(BOUNDS)
    p = np.clip(particles.x, np.array([0, 0]), BOUNDS - 1)
    img[p[:, 0], p[:, 1]] = 255
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img

for i in range(100):
    img = draw(particles)
    cv2.imshow("game", img)
    cv2.waitKey(100)
    simulation.step()