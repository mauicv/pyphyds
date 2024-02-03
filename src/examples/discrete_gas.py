from pyphyds.physics.particles.discrete_particles import DiscreteParticles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.laws.boundaries import DiscreteBoundaryBox
from pyphyds.physics.interactions.collision_interaction import DiscreteCollisionInteraction
from pyphyds.physics.simulation import Simulation
import cv2
import numpy as np


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 200
BOUNDS = np.array((SIZE, SIZE))
NUM_PARTICLES = 100

particles = DiscreteParticles(NUM_PARTICLES, BOUNDS, 2)
particle_map = ParticleMap(particles, [1], [1])

simulation = Simulation(
    particles=particles,
    particle_map=particle_map,
    laws=[
        DiscreteBoundaryBox(BOUNDS, [particles])
    ],
    interactions=[
        DiscreteCollisionInteraction(key_a=1, key_b=1, particle_map=particle_map)
    ]
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