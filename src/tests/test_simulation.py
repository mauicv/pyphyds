from pyphyds.simulation import Simulation
from pyphyds.particles import Particles
from pyphyds.laws.box_boundary import BoxBoundaryLaw


def test_simulation():
    particles = Particles(10, {'mass': 1.0})
    boundary_law = BoxBoundaryLaw(particles, -1.0, 1.0, -1.0, 1.0)
    sim = Simulation(
        particles=[particles],
        laws=[boundary_law],
        interactions=[]
    )
    sim.step(0.1)