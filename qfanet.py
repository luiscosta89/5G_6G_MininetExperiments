import numpy as np
import time

class QFANET:
    def __init__(self, num_nodes, update_frequency=0.1, expiration_time=0.3, learning_rate=0.6, discount_factor=1, epsilon=0.1):
        self.num_nodes = num_nodes
        self.q_table = np.zeros((num_nodes, num_nodes))  # Initialize Q-table
        self.r_table = np.zeros((num_nodes, num_nodes))  # Initialize Reward Table
        self.neighbor_tables = {i: {} for i in range(num_nodes)}  # Initialize neighbor tables for each node
        self.update_frequency = update_frequency  # 100 ms update frequency
        self.expiration_time = expiration_time  # 300 ms expiration time for neighbors
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.penalty_value = -100  # Penalty for routing holes and Not-ACK
        self.reward_max = 100  # Maximum reward for reaching destination
        self.reward_general = 50  # General reward for valid link
        self.distances = np.random.uniform(50, 300, (num_nodes, num_nodes))  # Simulate distances

    def send_hello_packet(self, node):
        """Simulate sending a HELLO packet from node to its neighbors."""
        packet = {
            'node': node,
            'position': np.random.uniform(0, 100),  # Simulated geographic location
            'energy': np.random.uniform(50, 100),  # Simulated energy level
            'mobility_model': 'Random Waypoint',  # Example mobility model
            'queue_delay': np.random.uniform(0.01, 0.1),  # Simulated queuing delay
            'q_value': self.q_table[node]
        }
        return packet

    def receive_hello_packet(self, node, packet):
        """Update the neighbor table when a node receives a HELLO packet."""
        neighbor = packet['node']
        # Update the neighbor table for the node with the received information
        self.neighbor_tables[node][neighbor] = {
            'position': packet['position'],
            'energy': packet['energy'],
            'queue_delay': packet['queue_delay'],
            'q_value': packet['q_value'],
            'last_update': time.time()  # Store the time of the last update
        }

    def neighbor_discovery(self):
        """Periodically broadcast HELLO packets and update neighbor tables."""
        for node in range(self.num_nodes):
            hello_packet = self.send_hello_packet(node)
            # Broadcast HELLO packet to all other nodes
            for neighbor in range(self.num_nodes):
                if neighbor != node:
                    self.receive_hello_packet(neighbor, hello_packet)

        # Remove outdated neighbors
        self.expire_neighbors()

    def expire_neighbors(self):
        """Remove neighbors from the neighbor table if they have not sent an update within the expiration time."""
        current_time = time.time()
        for node in range(self.num_nodes):
            to_remove = []
            for neighbor, info in self.neighbor_tables[node].items():
                if current_time - info['last_update'] > self.expiration_time:
                    to_remove.append(neighbor)
            for neighbor in to_remove:
                del self.neighbor_tables[node][neighbor]

    def calculate_velocity(self, i, j, destination, delay):
        """Calculate the velocity constraint between nodes i and j with respect to the destination."""
        distance_i = self.distances[i][destination]  # d(i, D)
        distance_j = self.distances[j][destination]  # d(j, D)
        velocity = (distance_i - distance_j) / delay
        return velocity

    def choose_action(self, state, destination):
        """Choose the next action based on neighbors and the velocity constraint."""
        if np.random.rand() < self.epsilon:
            neighbors = list(self.neighbor_tables[state].keys())
            return np.random.choice(neighbors)  # Exploration
        else:
            # Exploitation: select the best neighbor based on Q-values and velocity constraint
            best_neighbor = None
            best_value = -float('inf')

            for neighbor in self.neighbor_tables[state].keys():
                delay = self.neighbor_tables[state][neighbor]['queue_delay']
                velocity = self.calculate_velocity(state, neighbor, destination, delay)

                # Apply the velocity constraint: prefer positive velocities (closer to the destination)
                if velocity > 0:
                    q_value = self.q_table[state][neighbor]
                    if q_value > best_value:
                        best_value = q_value
                        best_neighbor = neighbor

            if best_neighbor is None:
                return 'routing_hole'  # If no valid neighbor is found, return routing hole

            return best_neighbor

    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update: state-action Q-value update based on the reward received."""
        best_future_action = np.max(self.q_table[next_state])
        q_update = reward + self.discount_factor * best_future_action
        self.q_table[state, action] += self.learning_rate * (q_update - self.q_table[state, action])

    def apply_penalty(self, state, action):
        """Apply a penalty in cases of routing hole or Not-ACK scenario."""
        self.q_table[state, action] += self.learning_rate * (self.penalty_value - self.q_table[state, action])

    def get_best_route(self, start_node, end_node):
        """Get the best route based on learned Q-values, considering routing holes, velocity constraints, and penalties."""
        current_node = start_node
        route = [current_node]
        while current_node != end_node:
            next_action = self.choose_action(current_node, end_node)
            if next_action == 'routing_hole':
                self.apply_penalty(current_node, route[-2])
                break
            route.append(next_action)
            current_node = next_action
        return route
