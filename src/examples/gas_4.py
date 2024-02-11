from pyphyds.physics.particles.particles import Particles
from pyphyds.physics.particles.particle_map import ParticleMap
from pyphyds.physics.laws.boundaries import BoxBoundary
from pyphyds.physics.interactions.force_interaction import ForceInteraction, AngularForceInteraction
from pyphyds.physics.interactions.local_interaction import StateTransitionInteraction, SpontaneousTransitionInteraction
from pyphyds.physics.interactions.collision_interaction import CollisionInteraction, SeparationInteraction
from pyphyds.physics.simulation import Simulation
import cv2
import numpy as np
import torch


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

SIZE = 500
# IMG_SIZE = 32
IMG_SIZE = 500
BOUNDS = torch.tensor([SIZE, SIZE])
NUM_PARTICLES = 300
PARTICLE_A_SIZE = 2
PARTICLE_B_SIZE = 2
PARTICLE_C_SIZE = 2
SPEED = 1

particles = Particles(NUM_PARTICLES, BOUNDS, SPEED)
particle_map = ParticleMap(
    particles, 
    4, [1/2, 1/2, 1/2, 1/2],
    properties={
        1: {
            'size': PARTICLE_A_SIZE,
            'color': (0, 0, 255),
            'is_active': 'True'
        },
        2: {
            'size': PARTICLE_B_SIZE,
            'color': (255, 0, 0),
            'is_active': 'True'
        },
        3: {
            'size': PARTICLE_C_SIZE,
            'color': (0, 255, 0),
            'is_active': 'True'
        },
        4: {
            'size': PARTICLE_C_SIZE,
            'color': (0, 255, 255),
            'is_active': 'True'
        }
    }
)

simulation = Simulation(
    particles=particles,
    particle_map=particle_map,
    laws=[
        BoxBoundary(BOUNDS, particles)
    ],
    interactions=[
        ForceInteraction(
            keys=[1, 2, 3, 4],
            particle_map=particle_map,
        ),
        # AngularForceInteraction(
        #     keys=[1, 2, 3, 4],
        #     particle_map=particle_map,
        # ),
        StateTransitionInteraction(
            source=1,
            catalyst=2,
            target=3,
            particle_map=particle_map
        ),
        StateTransitionInteraction(
            source=2,
            catalyst=1,
            target=0,
            particle_map=particle_map
        ),
        SpontaneousTransitionInteraction(
            source=3,
            targets=[2, 1],
            event_probability=0.01,
            particle_map=particle_map
        ),

        # CollisionInteraction(
        #     keys=[1, 2, 3, 4],
        #     particle_map=particle_map
        # ),
        # SeparationInteraction(
        #     keys=[1, 2, 3, 4],
        #     particle_map=particle_map
        # ),
    ]
)

def draw(particle_map):
    img = np.zeros((*BOUNDS.numpy(), 3))
    p = np.clip(particle_map.particles.x, np.array([0, 0]), BOUNDS - 1)
    for ind, x in enumerate(p):
        x = (int(x.numpy()[0]), int(x.numpy()[1]))
        properties = particle_map.get_properties(ind)
        color = (255, 255, 255)
        if properties['is_active']:
            color = properties['color']
        cv2.circle(img, x, int(properties['size']), color, -1)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    return img

while True:
    img = draw(particle_map)
    cv2.imshow("game", img)
    cv2.waitKey(1)
    simulation.step()