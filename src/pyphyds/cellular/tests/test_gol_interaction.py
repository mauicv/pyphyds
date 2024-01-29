from pyphyds.cellular.interactions.game_of_life import GameOfLife
from pyphyds.cellular import CellularSimulation
import torch


def test_cellular_automaton():
    cell_sim = CellularSimulation(
        width=3,
        height=3,
        num_states=1,
        num_histories=1,
        interaction_rules=[GameOfLife()]
    )
    cell_sim.states = cell_sim.states.new_zeros(1, 3, 3)
    cell_sim.step()
    assert cell_sim.states.shape == (1, 3, 3)
    assert torch.all(cell_sim.states == torch.tensor([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    ]))

