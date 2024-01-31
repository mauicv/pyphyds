from pyphyds.physics.particles import Particles, ParticleMap
from pyphyds.physics.laws.boundary_box import BoundaryBox
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction
from pyphyds.physics.simulation import Simulation
import cv2
import numpy as np
import torch


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 200
BOUNDS = torch.tensor([SIZE, SIZE])
NUM_PARTICLES = 10
SPEED = 2

particles = Particles(NUM_PARTICLES, BOUNDS, SPEED)
particle_map = ParticleMap(
    particles, [1], [1], properties={1: {'size': 2}}
)

simulation = Simulation(
    particles=particles,
    particle_map=particle_map,
    laws=[
        BoundaryBox(BOUNDS, [particles])
    ],
    interactions=[
        CollisionInteraction(
            key_a=1,
            key_b=1,
            particle_map=particle_map
        )
    ]
)

def draw(particles):
    img = np.zeros(BOUNDS.numpy())
    p = np.clip(particles.x, np.array([0, 0]), BOUNDS - 1)
    for x in particles.x:
        x = (int(x.numpy()[0]), int(x.numpy()[1]))
        cv2.circle(img, x, 4, (255, 0, 0), -1)
    # img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img

for i in range(100):
    img = draw(particles)
    cv2.imshow("game", img)
    cv2.waitKey(100)
    simulation.step()