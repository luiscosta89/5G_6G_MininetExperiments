import torch
import numpy as np
import random
import time
import json
import re
from mn_wifi.net import Mininet_wifi
from mn_wifi.node import OVSKernelAP
from mn_wifi.cli import CLI
from mn_wifi.wmediumdConnector import interference
from Ddpg_Routing_Fanet import DDPGAgent, ddpg_routing  # Import DDPG module

# Initialize DDPG agent
state_dim = 4  # Example state dimensions (position, average signal strength, queue length, etc.)
action_dim = 3  # Example action space (choosing among 3 next hops)
agent = DDPGAgent(state_dim, action_dim)

UDP_PORT = 12345
PACKET_SIZE = 512

def video_traffic_model():
    """Generate packet size and inter-arrival time based on a 3GPP video traffic model."""
    avg_packet_size = 1024  # Average size of packets in bytes
    packet_size_variation = 256  # Variability in packet size
    avg_inter_arrival_time = 0.03  # Average inter-arrival time in seconds (30ms)
    
    packet_size = int(random.normalvariate(avg_packet_size, packet_size_variation))
    packet_size = max(packet_size, 64)  # Ensure packet size is positive
    inter_arrival_time = random.expovariate(1 / avg_inter_arrival_time)
    
    return packet_size, inter_arrival_time

from math import sqrt

def select_cluster_heads_by_signal_strength(stations, cluster_info):
    cluster_heads = {}

    for cluster_id, station_names in cluster_info.items():
        best_score = float('-inf')
        selected_cluster_head = None

        for sta_name in station_names:
            sta = next(s for s in stations if s.name == sta_name)
            total_signal_strength = 0
            total_distance = 0

            # Calculate the sum of signal strength and distances to all other stations in the cluster
            for other_sta_name in station_names:
                if other_sta_name == sta_name:
                    continue

                other_sta = next(s for s in stations if s.name == other_sta_name)
                
                # Get signal strength
                signal_strength = sta.wintfs[0].get_rssi(other_sta.wintfs[0], 0)
                total_signal_strength += signal_strength
                
                # Calculate distance
                distance = sqrt((sta.params['position'][0] - other_sta.params['position'][0]) ** 2 +
                                (sta.params['position'][1] - other_sta.params['position'][1]) ** 2)
                total_distance += distance

            # Normalize and compute the score: (signal_strength - distance)
            # Higher signal strength and lower distance are better
            if len(station_names) > 1:
                avg_signal_strength = total_signal_strength / (len(station_names) - 1)
                avg_distance = total_distance / (len(station_names) - 1)
            else:
                avg_signal_strength = total_signal_strength
                avg_distance = total_distance

            # Score based on both average signal strength and distance
            score = avg_signal_strength - avg_distance

            if score > best_score:
                best_score = score
                selected_cluster_head = sta

        cluster_heads[cluster_id] = selected_cluster_head

    return cluster_heads


def get_average_signal_strength(station, neighbors):
    """Calculate the average signal strength of a station relative to its neighbors."""
    total_signal_strength = 0
    for neighbor in neighbors:
        total_signal_strength += station.wintfs[0].get_rssi(neighbor.wintfs[0], 0)
    return total_signal_strength / len(neighbors) if neighbors else -100  # Default to low signal strength if no neighbors

def get_queue_length(station):
    """Estimate the queue length of a station by checking its buffer size."""
    try:
        output = station.cmd('tc -s qdisc show dev %s' % station.wintfs[0].name)
        match = re.search(r'backlog (\d+)b', output)
        if match:
            return int(match.group(1)) / 1000  # Convert bytes to kilobytes
    except Exception as e:
        print(f"Error retrieving queue length for {station.name}: {e}")
    return 0  # Default to zero if unable to determine

def select_next_hop(station, neighbors):
    """Use DDPG to select the best next hop based on signal strength and queue length."""
    avg_signal_strength = get_average_signal_strength(station, neighbors)
    queue_length = get_queue_length(station)
    state = np.array([station.position[0], station.position[1], avg_signal_strength, queue_length])
    next_hop = ddpg_routing(agent, state, neighbors)
    return next_hop

def load_clusters_from_json(json_file):
    """Load cluster information from a JSON file."""
    with open(json_file, 'r') as file:
        clusters = json.load(file)
    return clusters

def udp_experiment(net, duration, clusters):
    """Simulate UDP traffic using video traffic model with clustering."""
    start_time = time.time()
    episode_rewards = []
    while time.time() - start_time < duration:
        for cluster_id, station_names in clusters.items():
            cluster_stations = [net[s] for s in station_names]
            cluster_head = select_next_hop(cluster_stations[0], cluster_stations[1:])
            
            for sta in cluster_stations:
                if sta == cluster_head:
                    continue  # Cluster head does not send to itself
                
                packet_size, inter_arrival_time = video_traffic_model()
                start_transmission = time.time()
                print(f"{sta.name} sending {packet_size} bytes to Cluster Head {cluster_head.name}")
                
                try:
                    with cluster_head.pexec(f'nc -lu {UDP_PORT}'):  # Simulate receiving at cluster head
                        sta.cmd(f'echo "X" | nc -u {cluster_head.IP()} {UDP_PORT}')
                except Exception as e:
                    print(f"Error in UDP transmission: {e}")
                
                end_transmission = time.time()
                transmission_delay = (end_transmission - start_transmission) * 1000  # Convert to ms
                print(f"Transmission delay: {transmission_delay:.2f} ms")
                
                # Store reward for DDPG performance evaluation
                reward = -transmission_delay  # Minimize delay
                episode_rewards.append(reward)
                avg_reward = np.mean(episode_rewards)
                print(f"DDPG Avg Reward: {avg_reward:.2f}")
                
                time.sleep(inter_arrival_time)

def topology():
    """Create and run a Mininet-WiFi topology."""
    net = Mininet_wifi(controller=None, accessPoint=OVSKernelAP)
    
    # Create stations
    stations = [net.addStation(f'sta{i+1}', position=f'{np.random.randint(0, 500)},{np.random.randint(0, 500)},0') for i in range(25)]
    
    # Create access points
    ap1 = net.addAccessPoint('ap1', ssid='ssid', mode='g', channel='1', position='250,250,0')
    
    net.configureWifiNodes()
    net.setPropagationModel(model="logDistance", exp=4.5)
    net.build()
    ap1.start([])
    
    # Load clusters from JSON
    clusters = load_clusters_from_json('clusters.json')
    
    # Start UDP experiment
    udp_experiment(net, duration=60, clusters=clusters)  # Run for 60 seconds
    
    CLI(net)
    net.stop()

if __name__ == '__main__':
    topology()
