from pyphyds.cellular.interactions.game_of_life import GameOfLife
from pyphyds.cellular import CellularSimulation
import cv2
import numpy as np


cv2.namedWindow("game", cv2.WINDOW_NORMAL)

cell_sim = CellularSimulation(
    width=100,
    height=100,
    num_states=2,
    num_histories=1,
    interaction_rules=[GameOfLife()]
)

for i in range(100):
    img = np.float32(cell_sim.states[0]) * 255
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("game", img)
    cv2.waitKey(100)
    cell_sim.step()