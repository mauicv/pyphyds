from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.laws.boundaries import TorusBoundary
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction, SeparationInteraction
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
    particles, 1, [1.], properties={1: {'size': 5}}
)

simulation = Simulation(
    particles=particles,
    particle_map=particle_map,
    laws=[
        TorusBoundary(BOUNDS, particles)
    ],
    interactions=[
        CollisionInteraction(
            keys=[1],
            particle_map=particle_map
        ),
        SeparationInteraction(
            keys=[1],
            particle_map=particle_map
        )
    ]
)

def draw(particles):
    img = np.zeros(BOUNDS.numpy())
    p = np.clip(particles.x, np.array([0, 0]), BOUNDS - 1)
    for x in particles.x:
        x = (int(x.numpy()[0]), int(x.numpy()[1]))
        cv2.circle(img, x, 5, (255, 0, 0), -1)
    # img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img

while True:
    img = draw(particles)
    cv2.imshow("game", img)
    cv2.waitKey(100)
    simulation.step()