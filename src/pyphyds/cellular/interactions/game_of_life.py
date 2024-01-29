from pyphyds.cellular.interactions.base_interaction import InteractionRuleBase
import torch


class GameOfLife(InteractionRuleBase):
    def __init__(self):
        super().__init__("GameOfLife")
        self.weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3).float()

    def __call__(self, states):
        with torch.no_grad():
            states = states.float()
            conv = torch.nn.functional.conv2d(states, self.weights, padding=1)
            states = (conv == 3) | (conv == 12) | (conv == 13)
        return states