from pyphyds.physics.laws.box_boundary import BoxBoundaryLaw
from pyphyds.physics.interactions.collision_law import CollisionLaw
from pyphyds.physics.particles import Particles
from pyphyds.physics import PhysicsSimulation
import torch
# Example file showing a basic pygame "game loop"
import pygame


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

# simulation setup
particles_1 = Particles(4, properties={'size': 25, 'color': "black"})
particles_2 = Particles(100, properties={'size': 5, 'color': "red"})
boundary_law_1 = BoxBoundaryLaw(
    particles_1, 0, screen.get_width(), 0, screen.get_height(),
)
boundary_law_2 = BoxBoundaryLaw(
    particles_2, 0, screen.get_width(), 0, screen.get_height(),
)
collision_law_11 = CollisionLaw([particles_1, particles_1])
collision_law_12 = CollisionLaw([particles_1, particles_2])
collision_law_22 = CollisionLaw([particles_2, particles_2])
sim = PhysicsSimulation(
    particles=[particles_1, particles_2],
    laws=[boundary_law_1, boundary_law_2],
    interactions=[collision_law_11, collision_law_12, collision_law_22]
)

def draw_particles(particles):
    for particle in particles.x:
        particle_pos = pygame.Vector2(particle[0], particle[1])
        pygame.draw.circle(
            screen,
            particles.properties['color'],
            particle_pos,
            particles.properties['size']
        )


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # run simulation
    sim.step(0.1)
    draw_particles(particles_1)
    draw_particles(particles_2)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()