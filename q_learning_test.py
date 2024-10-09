import networkx as nx
import numpy as np
import time, random
import matplotlib.pyplot as plt

def plot_graph_with_path(graph, optimal_path, start_node, goal_node):
    pos = nx.spring_layout(graph)  # You can use a different layout algorithm if needed

    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)

    # Highlight the optimal path
    edge_labels = {(u, v): f'{graph[u][v]["weight"]}' for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    nx.draw_networkx_nodes(graph, pos, nodelist=[start_node, goal_node], node_color='green', node_size=700)

    edge_list = [(optimal_path[i], optimal_path[i + 1]) for i in range(len(optimal_path) - 1)]
    
    # Draw all edges with thinner lines
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='gray', width=1)
    
    # Draw the edges in the optimal path with thicker red lines
    nx.draw_networkx_edges(graph, pos, edgelist=edge_list, edge_color='red', width=2)

    plt.title('Graph with Optimal Path')
    plt.show()

def plot_random_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=800, font_size=8)
    edge_labels = {(u, v): graph[u][v]['weight'] for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()

# Generate a random graph with weighted edges
def create_random_graph(num_nodes, seed):
    #random_graph = nx.erdos_renyi_graph(num_nodes, seed)
    random_graph = nx.gnp_random_graph(num_nodes, seed)
    random_graph.add_weighted_edges_from([(u, v, np.random.randint(1, 10)) for u, v in random_graph.edges()])

    return random_graph

def initialize_q_values(graph):
    q_values = {}
    for edge in graph.edges():
        q_values[edge] = 0
    return q_values

def choose_action(possible_actions, q_values, epsilon):
    valid_actions = [action for action in possible_actions if action in q_values]
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(possible_actions)
    else:
        max_q_value = max(q_values.get(action, 0) for action in possible_actions)
        best_actions = [action for action in possible_actions if q_values.get(action, 0) == max_q_value]
        return np.random.choice(best_actions)

def update_q_value(q_values, state, action, reward, next_state, alpha, gamma, graph):
    old_q_value = q_values.get((state, action), 0)
    best_future_q = max(q_values.get((next_state, next_action), 0) for next_action in graph.neighbors(next_state))
    new_q_value = old_q_value + alpha * (reward + gamma * best_future_q - old_q_value)
    q_values[(state, action)] = new_q_value

# def calculate_combined_reward(source, neighbor, current_latency, target_latency, latency_weight=0.5):
#     # Define your destination node or target
#     destination = 3  # Replace with your actual destination node

#     # Calculate the existing reward
#     if neighbor == destination:
#         existing_reward = 100
#     else:
#         neighbor_distance = calculate_distance(neighbor, destination)  # Define your distance calculation function
#         source_distance = calculate_distance(source, destination)
#         all_neighbors_farther = all(calculate_distance(n, destination) > source_distance for n in neighbor_table[source][0])

#         if all_neighbors_farther:
#             existing_reward = -100
#         else:
#             existing_reward = 50

#     # Calculate the latency reward using the function I provided earlier
#     latency_reward = network_latency_reward(current_latency, target_latency)

#     # Combine the existing reward and latency reward with a weighted sum
#     combined_reward = (1 - latency_weight) * existing_reward + latency_weight * latency_reward

#     return combined_reward

def network_latency_reward(current_latency, target_latency):
    # Define a threshold for acceptable latency
    latency_threshold = 100  # Adjust this threshold according to your specific use case

    # Calculate the difference between current and target latency
    latency_difference = current_latency - target_latency

    if current_latency <= latency_threshold:
        # Reward for latency below the threshold
        reward = 1.0
    else:
        # Penalize for latency above the threshold
        reward = 1.0 / (1.0 + abs(latency_difference))

    return reward

# def calculate_distance(node1, node2):
#     # Define your distance calculation logic (e.g., Euclidean distance)
#     # This is just a placeholder function, replace it with your actual implementation
#     return abs(node1 - node2)  # Placeholder distance value

def calculate_reward(graph, state, action, goal_node, penalty_nodes, q_values):
    # if action == goal_node:
    #     return 100  # Positive reward for reaching the goal
    # else:
    #     return -1 * graph[state][action]['weight']  # Negative reward based on edge weight
    if action == goal_node:
        return 100  # Positive reward for reaching the goal
    else:
        # Check if the action is a local minimum (routing hole)
        neighbors = list(graph.neighbors(action))
        distances = [graph[action][neighbor]['weight'] for neighbor in neighbors]
        if all(distance >= graph[state][action]['weight'] for distance in distances):
            # Routing hole detected, apply penalty
            penalty_nodes.add(action)
            return -100
        else:
            # Check if Not-ACK condition is met
            if action not in penalty_nodes:
                return -1 * graph[state][action]['weight']  # Negative reward based on edge weight
            else:
                # Penalty mechanism: Minimum reward for the link and update Q-value
                reward = -1 * graph[state][action]['weight']
                update_q_value(q_values, state, action, reward, action, 0.1, 0.9, graph)
                return reward


def q_learning(graph, start_node, goal_node, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    start_time = time.time()
    q_values = initialize_q_values(graph)
    episode_history = []
    penalty_nodes = set()
    
    for episode in range(num_episodes):
        state = start_node
        episode_steps = []

        while state != goal_node:
            possible_actions = list(graph.neighbors(state))
            action = choose_action(possible_actions, q_values, epsilon)
            
            # Check if the chosen action is valid
            if action not in possible_actions:
                break
            
            next_state = action
            reward = calculate_reward(graph, state, action, goal_node, penalty_nodes, q_values)
            #reward = -graph[state][action]['weight']  # Invert the weight as we want to minimize the cost
            #update_q_value(q_values, state, action, reward, next_state, alpha, gamma, graph)
            episode_steps.append((state, action, reward, next_state))
            state = next_state

        episode_history.append(episode_steps)
        if len(episode_history) > 10:
            episode_history.pop(0)  # Remove the oldest episode

        for i, episode_steps in enumerate(reversed(episode_history)):
            for step in reversed(episode_steps):
                state, action, reward, next_state = step
                update_q_value(q_values, state, action, reward, next_state, alpha, gamma, graph)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Q-values: {q_values}")
    
    end_time = time.time()
    
    # Print final Q-value table
    print("Final Q-value table:")
    for key, value in q_values.items():
        print(f"{key}: {value}")
    
    # Extract the optimal path from the learned Q-values
    optimal_path = [start_node]
    current_node = start_node
    while current_node != goal_node:
        possible_actions = list(graph.neighbors(current_node))
        best_action = max(possible_actions, key=lambda action: q_values.get((current_node, action), 0))
        optimal_path.append(best_action)
        current_node = best_action
    
    print("Optimal path from ", start_node, " to ", goal_node, ": ", optimal_path)
    print("Execution time:", end_time - start_time, "seconds")

    return optimal_path

# # Example usage:
# G = nx.Graph()
# # Tuplas: (nodo1 (int ou str), nodo2 (int ou str), peso (int ou float))
# G.add_weighted_edges_from([(0, 1, 4), 
#                            (1, 2, 8), 
#                            (2, 3, 7), 
#                            (3, 4, 9),
#                            (4, 5, 10),
#                            (5, 6, 2), 
#                            (6, 7, 1),
#                            (7, 0, 8),
#                            (7, 1, 11),
#                            (7, 8, 7),
#                            (8, 2, 2),
#                            (2, 5, 4),
#                            (6, 8, 6),
#                            (3, 5, 14)])


# # Example usage
# num_nodes = 10
# num_edges = 15
# new_G = create_and_display_random_weighted_graph(num_nodes, num_edges)

# start_node = 0
# goal_node = 8

#optimal_path = q_learning(G, start_node, goal_node)
#plot_graph_with_path(G, optimal_path, start_node, goal_node)

# Generate a random graph with weighted edges
#random_graph = nx.erdos_renyi_graph(15, 0.3)
#random_graph.add_weighted_edges_from([(u, v, np.random.randint(1, 10)) for u, v in random_graph.edges()])

random_G = create_random_graph(25, 0.25)

# start_node = np.random.choice(list(random_G.nodes()))
# goal_node = np.random.choice(list(random_G.nodes()))

start_node = 1
goal_node = 11

print("Start node: ", start_node)
print("Goal node: ", goal_node)

optimal_path = q_learning(random_G, start_node, goal_node, num_episodes=5000, alpha=0.2, gamma=0.9, epsilon=0.2)
plot_graph_with_path(random_G, optimal_path, start_node, goal_node)





