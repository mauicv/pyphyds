from pyphyds.laws.box_boundary import BoxBoundaryLaw
from pyphyds.particles import Particles
from pyphyds.simulation import Simulation
import torch
# Example file showing a basic pygame "game loop"
import pygame


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

# simulation setup
particles = Particles(1)
boundary_law = BoxBoundaryLaw(
    particles, 'box', 0, screen.get_width(), 0, screen.get_height(),
)
sim = Simulation(
    particles=[particles],
    laws=[boundary_law],
    interactions=[]
)

def draw_particles(particles):
    for particle in particles.x:
        particle_pos = pygame.Vector2(particle[0], particle[1])
        pygame.draw.circle(screen, "black", particle_pos, 5)


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # run simulation
    sim.step(2)
    draw_particles(particles)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()