from pyphyds.cellular import CellularSimulation


def test_cellular_automaton():
    cell_sim = CellularSimulation(
        width=10,
        height=10,
        num_states=2,
        num_histories=3
    )
    assert cell_sim.states.shape == (3, 10, 10)
    assert cell_sim.states.max() < 2
    assert cell_sim.states.min() >= 0