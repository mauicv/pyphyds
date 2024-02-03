from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.laws.boundaries import BoundaryBox
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction
from pyphyds.physics.simulation import Simulation
import cv2
import numpy as np
import torch


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 500
IMG_SIZE = 32
BOUNDS = torch.tensor([SIZE, SIZE])
NUM_PARTICLES = 25
PARTICLE_A_SIZE = 10
PARTICLE_B_SIZE = 20
SPEED = 2

particles = Particles(NUM_PARTICLES, BOUNDS, SPEED)
particle_map = ParticleMap(
    particles, 
    [1, 2],
    [0.9, 0.1],
    properties={
        1: {
            'size': PARTICLE_A_SIZE,
            'color': (0, 0, 255)
        },
        2: {
            'size': PARTICLE_B_SIZE,
            'color': (255, 0, 0)
        }
    }
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
            key_b=2,
            particle_map=particle_map
        )
    ]
)

def draw(particle_map):
    img = np.zeros((*BOUNDS.numpy(), 3))
    p = np.clip(particle_map.particles.x, np.array([0, 0]), BOUNDS - 1)
    for ind, x in enumerate(p):
        x = (int(x.numpy()[0]), int(x.numpy()[1]))
        properties = particle_map.get_properties(ind)
        cv2.circle(img, x, int(properties['size']), properties['color'], -1)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img

while True:
    img = draw(particle_map)
    cv2.imshow("game", img)
    cv2.waitKey(1)
    simulation.step()