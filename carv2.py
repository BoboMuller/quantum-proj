import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from overdrive import Overdrive
import time

location = 0
peice = 0
offset = 0
speed = 0
clockwise = 0
peiceIdx = 0

prevPeice = 0
prevLocation = 0


def locationChangeCallback(addr, data):

    global location 
    global peice 
    global offset 
    global speed 
    global clockwise

    location = data["location"]
    peice = data["piece"]
    offset = data["offset"]
    speed = data["speed"]
    clockwise = data["clockwise"]

    print(location, peice, offset, speed, clockwise)

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


# Define your Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def save_model(model, is_target_network):
    t = time.localtime()
    st = f"{t[2]}{t[3]}{t[4]}"
    if is_target_network:
        torch.save(model, f"target_{st}.pth")
    else:
        torch.save(model, f"q_{st}.pth")

def load_model(path_as_string):
    # Usage:
    # q_network = load_model("q_91039.pth")
    # target_network = load_model("target_91039.pth")

    model = torch.load(path_as_string)
    if path_as_string[0] == "t":
        model.eval()
    return model

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


def classifyTrack(peice):

    corners = [17, 18, 20, 23]
    straight = [39, 40, 43, 46]

     
    if peice in corners:
        return 1
    else:
        return 0
    
def nextTrack():
    global peice
    global prevPeice
    global location
    global prevLocation
    
    global peiceIdx

    trackList = [33, 23, 23, 20, 17, 10, 40, 40, 40, 18, 23, 39, 39, 36, 17, 10, 20, 39, 40, 18, 39, 40, 20, 39, 46, 43, 34]  
    corners = [17, 18, 20, 23]
    straight = [39, 40, 43, 46]

    if prevPeice != peice or location < prevLocation:
        peiceIdx += 1

    if trackList[peiceIdx - 1] not in [46, 34, 43]:
        nxtTrack = [classifyTrack(trackList[peiceIdx + 1 // len(trackList)]), classifyTrack(trackList[peiceIdx + 2 // len(trackList)]), classifyTrack(trackList[peiceIdx + 3 // len(trackList)])]
    else:
        nxtTrack = [0, 0, 0]

    return nxtTrack


def perform_action(action):
    # Here we need some code to map an action to the API 
    # Also returns a done flag when the lap is finished 
    # Also returns a terminated flag if the model tried it for too long and did not finish the lap (too slow or whatever reason)
    # Another very IMPORTANT thing is to perform normalization on the outsputs that the car gives us. In the best case the data ranges from 0 to 1 or -1 to 1

    global prevPeice 
    global location 
    global peice 
    global offset 
    global speed 
    global clockwise

    global peiceIdx

    
    maxSpeed = 1000
    deltaSpeed = int(0.15 * maxSpeed)
    speedApply = 0
    
    if action:
        speedApply = speed + deltaSpeed
    elif speed > 100:
        speedApply = speed - 100

    carConnect(speedApply)
    
    done = False

    print("Peice: ", peice, " Last Peice: ", prevPeice)
    print("Track: ", nextTrack())

    trackNext = nextTrack()

    #write a lap completion logic and termination logic.
    if (peice == 33) and (prevPeice == 39 or prevPeice == 34 or prevPeice == 46 or prevPeice == 43):
        done = True
        peiceIdx = 0
        print(done)
    
        
    
    prevPeice = peice
    prevLocation = location

    time.sleep(0.8)
    return (location, peice, offset, speed, trackNext[0], trackNext[1], trackNext[2]), done 

def calculate_reward(speed, lapTime,  done = False, terminated = False):

    rcont = speed / 1000
    rlap = 0
    if terminated:
        rcont = 0 
    elif done:
        rlap = 500 / lapTime

    r = rcont + rlap
    return r
    
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
state_size = 7 # Number of features we consider from the measurements
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
epsilon_decay = 0.9975 # decay rate for exploration probability. Increase to 0.9985 for final training
update_target_freq = 10 # Longer update timeframes leads to faster convergence, but have drawbacks of course...
save_freq = 5

replay_buffer = ReplayBuffer(buffer_capacity)
epsilon = epsilon_start



for episode in range(num_episodes):
    # Here we need a default state for the car, maybe we can do it somehow nicer
    state = np.array([0, 0, 0, 0, 0, 0, 0], dtype="float32")
    sum_reward = 0
    done = False
    lapStartTime = time.time()
    print("--------->", episode)
    while not done:
        # Choose action using epsilon-greedy strategy
        q_values = q_network(torch.tensor(state, dtype=torch.float32))
        action = epsilon_greedy_action(q_values, epsilon)

        # Take action, observe next state and reward
        next_state, done = perform_action(action)
        
        print(done, peice)
        currTime = time.time()
        lapTime = 0
        if done:
            lapTime = currTime - lapStartTime
            
        reward = calculate_reward(next_state[3], lapTime,  done)

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
    if episode % save_freq == 0:
        save_model(q_network, False)
        save_model(target_network, True)

    # Print or log episode statistics
    print(f"Episode {episode + 1}, Lap reward: {sum_reward}, Lap Time: {lapTime}")
