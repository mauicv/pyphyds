from pyphyds.discrete_physics.particles import Particles, ParticleMap
from pyphyds.discrete_physics.laws.boundary_box import BoundaryBox
from pyphyds.discrete_physics.laws.randomness import RandomnessLaw
from pyphyds.discrete_physics.interactions.collision_interaction import CollisionInteraction
from pyphyds.discrete_physics.simulation import Simulation
import cv2
import numpy as np


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 50
BOUNDS = np.array((SIZE, SIZE))
NUM_PARTICLES = 50
SPEED = 1

particles = Particles(NUM_PARTICLES, BOUNDS, SPEED)

particles.v = np.zeros_like(particles.v)

particle_map = ParticleMap(particles, [1, 2], [0.1, 0.9])

simulation = Simulation(
    particles=particles,
    particle_map=particle_map,
    laws=[
        BoundaryBox(BOUNDS, [particles]),
        RandomnessLaw([particles])
    ],
    interactions=[
        CollisionInteraction(key_a=1, key_b=2, particle_map=particle_map),
    ]
)

colour_map = {
    1: (0, 0, 255),
    2: (255, 0, 0)
}

def draw(particle_map):
    img = np.zeros((*BOUNDS, 3), dtype=np.uint8)
    for cls in particle_map.classes:
        p_x = particle_map.get_class('x', cls)
        p_x = np.clip(p_x, np.array([0, 0]), BOUNDS - 1)
        img[p_x[:, 0], p_x[:, 1]] = colour_map[cls]
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img

while True:
    img = draw(particle_map=particle_map)
    cv2.imshow("game", img)
    cv2.waitKey(100)
    simulation.step()