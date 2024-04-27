import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from overdrive import Overdrive
import time

from carv2 import locationChangeCallback
from carv2 import QNetwork
from carv2 import load_model
from carv2 import epsilon_greedy_action
from carv2 import classifyTrack
from carv2 import nextTrack
from carv2 import perform_action

location = 0
peice = 0
offset = 0
speed = 0
clockwise = 0
peiceIdx = 0
prevPeice = 0
prevLocation = 0

cars = {
    'skull': 'FD:24:03:51:B8:44',
    'shock': 'D1:8A:9D:BB:B6:4F',
    'police': 'FF:B1:ED:98:FF:FF',
    'bigBang': 'F2:2D:0F:19:FD:27',
    'military' : 'D1:A3:08:F1:80:0D',
    'rt' : 'C9:1C:CE:BF:73:08'
}

car = Overdrive(cars['military'])
car.setLocationChangeCallback(locationChangeCallback)
car.changeSpeed(0, 0)

def demo():
    q_network = load_model("q_191341.pth")
    state = np.array([0, 0, 0, 0, 0, 0, 0], dtype="float32")
    done = False
    lap_start = time.time()
    while not done:
        q_values = q_network(torch.tensor(state, dtype=torch.float32))
        action = epsilon_greedy_action(q_values, 0)
        next_state, done = perform_action(action)
        state = next_state
    lap_end = time.time()
    lapTime = lap_end - lap_start
    print(f"Lap Time: {lapTime}")

demo()