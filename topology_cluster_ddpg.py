import torch, threading, sys
import numpy as np
import matplotlib.pyplot as plt
import random, time, json, re, csv
from mn_wifi.net import Mininet_wifi
from mn_wifi.node import OVSKernelAP
from mn_wifi.cli import CLI
from mn_wifi.wmediumdConnector import interference
from ddpg import DDPGAgent, ddpg_routing  # Import DDPG module

# Initialize DDPG agent
state_dim = 4  # Example state dimensions (position, average signal strength, queue length, etc.)
action_dim = 3  # Example action space (choosing among 3 next hops)
agent = DDPGAgent(state_dim, action_dim)

hosts = []
stations = []
num_stations_ap1 = 13
num_stations_ap2 = 12
num_stations_total = 25

UDP_PORT = 12345
PACKET_SIZE = 512

def save_stations_positions(stations):
    """Save the positions of stations to a CSV file."""
    # File name where the positions will be stored
    output_file = "node_positions.csv"

    # Writing positions to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optionally write a header
        writer.writerow(["x", "y", "z"])
        # Write each position
        for sta in stations:
            writer.writerow(sta.position)

    print(f"Node positions saved to {output_file}")

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
                distance = sqrt((sta.position[0] - other_sta.position[0]) ** 2 +
                                (sta.position[1] - other_sta.position[1]) ** 2)
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

def start_udp_listener(node, port=UDP_PORT):
    # Kill any old listeners on this port for cleanliness
    node.cmd(f'pkill -f "nc -l.*u.* {port}"')
    # Start a persistent UDP listener (-k) in the background
    node.cmd(f'nohup sh -c "nc -luk {port} > /dev/null 2>&1" &')

def udp_experiment(net, duration, clusters):
    """
    Run a UDP transmission experiment for all clusters.
    - net: Mininet-WiFi network
    - duration: experiment runtime (seconds)
    - clusters: dict {cluster_id: [station names]}
    """
    """ Simulate UDP traffic using video traffic model with clustering."""
    start_time = time.time()
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
                        print("enviando pacote")
                except Exception as e:
                    print(f"Error in UDP transmission: {e}")
                
                end_transmission = time.time()
                transmission_delay = (end_transmission - start_transmission) * 1000  # Convert to ms
                print(f"Transmission delay: {transmission_delay:.2f} ms")
                
                time.sleep(inter_arrival_time)


def plot_throughput(timestamps, throughput):
    # Calculate the average throughput
    avg_throughput = sum(throughput) / len(throughput)

    # Create a plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot throughput over time
    ax1.plot(timestamps, throughput, color='tab:blue', marker='o', label='Throughput')

    # Plot the average throughput line
    ax1.axhline(y=avg_throughput, color='red', linestyle='--', label=f'Average Throughput: {avg_throughput:.2f} bps')

    # Set labels and title
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Throughput (bps)')
    ax1.set_title('Throughput over Time')

    # Add legend
    ax1.legend(loc='upper right')

    # Display the plot
    plt.grid(True)
    plt.show()

def topology():
    """Create and run a Mininet-WiFi topology."""
    net = Mininet_wifi(controller=None, accessPoint=OVSKernelAP)
    
    # Create stations
    #stations = [net.addStation(f'sta{i+1}', position=f'{np.random.randint(0, 500)},{np.random.randint(0, 500)},0') for i in range(25)]

    for i in range(num_stations_total):
        station_name = 'sta{}'.format(i+1)
        station = net.addStation(station_name, position=f"{random.randint(10, 490)},{random.randint(10, 490)}, 0", mac='00:00:00:00:00:{}'.format(i+4), ip='10.0.0.{}/8'.format(i+4), txpower=70)
        stations.append(station)
    
    # Create access points
    #ap1 = net.addAccessPoint('ap1', ssid='ssid', mode='g', channel='1', position='250,250,0')

    # Create and connect hosts
    h1 = net.addHost('h1', mac='00:00:00:00:00:01', ip='10.0.0.1/8')
    h2 = net.addHost('h2', mac='00:00:00:00:00:02', ip='10.0.0.2/8')
    h3 = net.addHost('h3', mac='00:00:00:00:00:03', ip='10.0.0.3/8')

    hosts = []
    hosts.append(h1)
    hosts.append(h2)
    hosts.append(h3)    
    
    for i, station in enumerate(stations):
        if i < num_stations_ap1:
            net.addLink(station, h1)
        elif i < num_stations_ap1 + num_stations_ap2:
            net.addLink(station, h2)
        else:
            net.addLink(station, h3)
    
    print("*** Creating and Configuring Nodes ***")
    net.configureWifiNodes()
    net.setPropagationModel(model="logDistance", exp=4.5)
    print("*** Building network ***")
    net.build()
    
    # Load clusters from JSON
    clusters = load_clusters_from_json('clusters.json')
    
    # Start UDP experiment
    net.start()

    # Set the mobility model (You can choose another mobility model as needed)
    net.startMobility(time=0, model='RandomWayPoint', max_x=100, max_y=100, min_v=0.5, max_v=1)

    print("Starting experiment thread...")
    print(stations)

    # Start the experiment in a separate thread
    experiment_thread = threading.Thread(target=udp_experiment, args=(net, 100, clusters))
    experiment_thread.start()

    # Wait for the experiment to finish
    experiment_thread.join()
    
    #CLI(net)
    net.stop()

if __name__ == '__main__':
    topology()
