import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class QFANET_DQN_Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, lr=0.0005, buffer_capacity=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=buffer_capacity)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

def compute_velocity_constraint(src, dst, destination):
    d_src = np.linalg.norm(np.array(src.position[:2]) - np.array(destination.position[:2]))
    d_dst = np.linalg.norm(np.array(dst.position[:2]) - np.array(destination.position[:2]))
    link_distance = np.linalg.norm(np.array(src.position[:2]) - np.array(dst.position[:2]))
    velocity = (d_src - d_dst) / link_distance if link_distance != 0 else -1.0
    return velocity

def penalty_for_velocity(velocity):
    return -1.0 if velocity < 0 else 0.0

def calculate_latency_reward(current_latency, previous_latency, latency_threshold=100.0):
    if current_latency <= latency_threshold:
        return 1.0
    else:
        latency_difference = current_latency - previous_latency
        return 1.0 / (1.0 + abs(latency_difference))

def qfanet_dqn_routing(agent, state, src, neighbors, previous_latency, destination, latency_measure_func):
    action_index = agent.select_action(state)
    if action_index >= len(neighbors):
        action_index = len(neighbors) - 1
    selected_neighbor = neighbors[action_index]

    current_latency = latency_measure_func(src, selected_neighbor)
    velocity = compute_velocity_constraint(src, selected_neighbor, destination)
    velocity_penalty = penalty_for_velocity(velocity)

    reward = calculate_latency_reward(current_latency, previous_latency) + velocity_penalty

    next_state = np.array([
        selected_neighbor.position[0],
        selected_neighbor.position[1],
        np.mean([
            selected_neighbor.wintfs[0].get_rssi(n.wintfs[0], 0)
            for n in neighbors if n != selected_neighbor
        ]) if len(neighbors) > 1 else -100,
        0
    ])

    agent.store_experience(state, action_index, reward, next_state)
    agent.update_model()

    return selected_neighbor