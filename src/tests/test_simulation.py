from pyphydl.simulation import Simulation
from pyphydl.particles import Particles
from pyphydl.laws import BoxBoundaryLaw


def test_simulation():
    particles = Particles(10, {'mass': 1.0})
    boundary_law = BoxBoundaryLaw(particles, 'box', -1.0, 1.0, -1.0, 1.0)
    sim = Simulation(
        particles=[particles],
        laws=[boundary_law],
        interactions=[]
    )
    sim.step(0.1)