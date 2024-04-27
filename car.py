import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from overdrive import Overdrive


location = 0
peice = 0
offset = 0
speed = 0
clockwise = 0


def locationChangeCallback(addr, location):
    location = location["location"]
    peice = location["piece"]
    offset = location["offset"]
    speed = location["speed"]
    clockwise = location["clockwise"]

cars = {
    'skull': 'FD:24:03:51:B8:44',
    'shock': 'D1:8A:9D:BB:B6:4F'
        }

car = Overdrive(cars['skull'])
car.setLocationChangeCallback(locationChangeCallback)
car.changeSpeed(0, 0)








# Define your Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)





class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add_experience(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

def epsilon_greedy_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    else:
        return np.argmax(q_values.detach())

def carConnect(speed):
    #car.setLocationChangeCallback(locationChangeCallback)
    car.changeSpeed(speed, 250)


def perform_action(action, dummy_terminationcounter):
    # Here we need some code to map an action to the API 
    # Also returns a done flag when the lap is finished 
    # Also returns a terminated flag if the model tried it for too long and did not finish the lap (too slow or whatever reason)
    # Another very IMPORTANT thing is to perform normalization on the outsputs that the car gives us. In the best case the data ranges from 0 to 1 or -1 to 1
    
    #if action < 2:
    #    d = np.array([0, 9, 11, 0], dtype="float32")
    #else:
    #    d = np.array([0, 9, 12, 3], dtype="float32")
    #
    
    print(action)
    dummy_terminationcounter = dummy_terminationcounter - 1
    if dummy_terminationcounter <= 0:
        x = 1
    else:
        x = 0
    
    maxSpeed = 1000
    deltaSpeed = int(0.6 * maxSpeed)
    speedApply = 0
    
    if action:
        speedApply = speed + deltaSpeed
    elif speed > 100:
        speedApply = speed - 100
            
    #speedApply = speed + deltaSpeed if action == 1 else speed - deltaSpeed
    print(speedApply)
    carConnect(speedApply)
    
    termination = False
    done = False
    
    return (location, peice, offset, speed, clockwise), done, termination, dummy_terminationcounter 

def calculate_reward(next_state):
    
    return sum(next_state)
    
    # Here we calculate the reward based on the new state the car is in
    # For example, if it is now very close to the wall because it took the curve too fast the reward could be 0

def update_dqn(q_network, target_network, optimizer, batch, gamma):
    # Extract components from batch
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert components to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q-values for current and next states
    q_values = q_network(states)
    q_values_next = target_network(next_states).detach()

    # Compute target Q-values
    target_q_values = rewards + (1 - dones) * gamma * q_values_next.max(dim=1)[0]

    # Compute Q-values for the chosen actions
    q_values_selected = q_values.gather(1, actions.unsqueeze(1))

    # Compute the loss
    loss = nn.MSELoss()(q_values_selected.squeeze(), target_q_values)

    # Perform a gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize Q-network, target network, optimizer, and replay buffer
state_size = 5 # Number of features we consider from the measurements
action_size = 2  # Number of actions the car can execute (speed up by 10%, speed down by 10%)
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())  # Same network, but independend since we don't want to run behind a running target
target_network.eval()  # Avoid training

optimizer = optim.Adam(q_network.parameters(), lr=0.001)

buffer_capacity = 1000 # Size of the memory / circular buffer
buffer = np.zeros((buffer_capacity, 5), dtype="object")

num_episodes = 100  # specify the number of episodes
batch_size = 32  # specify the batch size
gamma = 0.99  # specify the discount factor
epsilon_start = 1.0  # starting exploration probability
epsilon_end = 0.01  # ending exploration probability
epsilon_decay = 0.995  # decay rate for exploration probability
update_target_freq = 10 # Longer update timeframes leads to faster convergence, but have drawbacks of course...

replay_buffer = ReplayBuffer(buffer_capacity)
epsilon = epsilon_start


for episode in range(num_episodes):
    # Here we need a default state for the car, maybe we can do it somehow nicer
    state = np.array([0, 10, 10, 0, 0], dtype="float32")
    sum_reward = 0
    done = False
    terminated = False
    dummy_terminationcounter = 100

    while not done and not terminated:
        # Choose action using epsilon-greedy strategy
        q_values = q_network(torch.tensor(state, dtype=torch.float32))
        action = epsilon_greedy_action(q_values, epsilon)

        # Take action, observe next state and reward
        next_state, done, terminated, dummy_terminationcounter = perform_action(action, dummy_terminationcounter)
        reward = calculate_reward(next_state)

        # Store experience in replay buffer
        experience = (state, action, reward, next_state, done)
        replay_buffer.add_experience(experience)

        if len(replay_buffer.buffer) > batch_size:
            # Sample random batch from replay buffer
            batch = replay_buffer.sample_batch(batch_size)
            batch = np.array(batch, dtype="object") # Performance reasons, nothing else

            # Update Q-network parameters using the DQN algorithm
            update_dqn(q_network, target_network, optimizer, batch, gamma)

            # Update exploration probability
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Move to the next state
        state = next_state
        sum_reward += reward
        
    if episode % update_target_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Print or log episode statistics
    print(f"Episode {episode + 1}, Lap reward: {sum_reward}")
