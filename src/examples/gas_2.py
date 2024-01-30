from pyphyds.physics.particles import Particles
from pyphyds.physics.laws.boundary_box import BoundaryBox
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction
from pyphyds.physics import Simulation
import cv2
import numpy as np


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 50
BOUNDS = np.array((SIZE, SIZE))
NUM_PARTICLES = 50

particles_a = Particles(NUM_PARTICLES, BOUNDS, 2)
particles_b = Particles(NUM_PARTICLES, BOUNDS, 2)

simulation = Simulation(
    particles=[particles_a, particles_b],
    laws=[BoundaryBox(BOUNDS, [particles_a, particles_b])],
    interactions=[CollisionInteraction(particles_a, particles_b)]
)


def draw(particles):
    img = np.zeros((*BOUNDS, 3), dtype=np.uint8)

    for particle, color in particles:
        p = np.clip(particle.x, np.array([0, 0]), BOUNDS - 1)
        img[p[:, 0], p[:, 1]] = color
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img


for i in range(100):
    img = draw(
        particles=[
            (particles_a, (255, 0, 0)),
            (particles_b, (0, 0, 255))
        ]
    )
    cv2.imshow("game", img)
    cv2.waitKey(100)
    simulation.step()
