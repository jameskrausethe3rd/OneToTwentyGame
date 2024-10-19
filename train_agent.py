import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from game_env import OneToTwentyGame

# Define constants
STATE_SIZE = 22  # Number of slots (20) + current_number (1) + turn (1)
ACTION_SIZE = 20  # Number of possible actions (slots)
GAMMA = 0.99  # Discount factor
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 25000
NUM_EPISODES = 500

# Define Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, experience, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), size=BATCH_SIZE, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority

# Initialize environment
env = OneToTwentyGame()

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, optimizer, and replay buffer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
optimizer = optim.Adam(model.parameters())
memory = PrioritizedReplayBuffer(MEMORY_SIZE)  # Use prioritized replay buffer

def choose_action(state, epsilon):
    if isinstance(state, dict):
        state_array = np.array(state['slots'] + [state['current_number']] + [state['turn']])
    else:
        raise TypeError("State should be a dictionary, but received a different type.")
        
    state_tensor = torch.FloatTensor(state_array).to(device)
    state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

    valid_spots = env.get_valid_spots()  # Get valid actions from the environment
    
    if random.random() < epsilon:
        # Choose randomly from valid spots
        valid_spots = [spot for spot in valid_spots if spot not in env.slots]
        if valid_spots:
            return random.choice(valid_spots)
        else:
            return None
    else:
        q_values = model(state_tensor).squeeze(0).cpu().detach().numpy()
        
        # Filter Q-values to valid spots only
        valid_q_values = [(spot, q_values[spot]) for spot in valid_spots]
        if valid_q_values:
            # Choose the spot with the highest Q-value
            return max(valid_q_values, key=lambda x: x[1])[0]
        else:
            return None

def replay():
    if len(memory.buffer) < BATCH_SIZE:
        return

    experiences, indices, weights = memory.sample(beta=0.4)

    states, actions, rewards, next_states, dones = zip(*experiences)

    # Convert state and next_state dictionaries to numpy arrays
    states_array = np.array([s['slots'] + [s['current_number']] + [s['turn']] for s in states])
    next_states_array = np.array([s['slots'] + [s['current_number']] + [s['turn']] for s in next_states])

    states_tensor = torch.FloatTensor(states_array).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    next_states_tensor = torch.FloatTensor(next_states_array).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)
    weights_tensor = torch.FloatTensor(weights).to(device)

    q_values = model(states_tensor)
    next_q_values = model(next_states_tensor).detach()

    q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    targets = rewards_tensor + GAMMA * torch.max(next_q_values, dim=1)[0] * (1 - dones_tensor)

    # Compute TD error (difference between Q-value and target)
    td_errors = (q_values - targets).detach().cpu().numpy()
    new_priorities = np.abs(td_errors) + 1e-6  # Ensure non-zero priorities

    # Update replay buffer priorities based on TD error
    memory.update_priorities(indices, new_priorities)

    # Calculate loss with weights from prioritized replay
    loss = (weights_tensor * (q_values - targets).pow(2)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_dqn():
    epsilon = 1.0
    highest_reward = 0
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = choose_action(state, epsilon)
            if action is None:
                done = True
                break
            next_state, reward, done = env.step(action)
            
            # Set initial priority based on reward (you can adjust this formula)
            priority = abs(reward) + 1.0  # Add a constant to ensure non-zero priority
            
            memory.push((state, action, reward, next_state, done), priority=priority)
            state = next_state
            total_reward += reward
            replay()
        
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        print(f"Episode: {episode+1}/{NUM_EPISODES}, Total Reward: {total_reward}, Epsilon: {epsilon:.6f}")

        if (total_reward > highest_reward):
            highest_reward = total_reward
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    print(f"Highest Reward: {highest_reward}")

if __name__ == "__main__":
    train_dqn()
