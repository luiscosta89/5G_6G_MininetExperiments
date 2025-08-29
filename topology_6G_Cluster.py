from mn_wifi.net import Mininet_wifi
from mn_wifi.node import Controller, OVSKernelAP
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from mn_wifi.propagationModels import PropagationModel
from mn_wifi.mobility import mobility
import random
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import re, csv
from math import sqrt
from dqfanet_routing import QFANET_DQN_Agent, qfanet_dqn_routing

num_stations_ap1 = 13
num_stations_ap2 = 12

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

def latency_measure_func(src, dst):
    result = src.cmd(f'ping -c1 -W1 {dst.IP()}')
    match = re.search(r'time=(\d+\.\d+)', result)
    if match:
        return float(match.group(1))
    else:
        return 999.0  # fallback if ping fails

def get_average_signal_strength(station, neighbors):
    total_signal_strength = 0
    for neighbor in neighbors:
        total_signal_strength += station.wintfs[0].get_rssi(neighbor.wintfs[0], 0)
    return total_signal_strength / len(neighbors) if neighbors else -100

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


def get_queue_length(station):
    try:
        output = station.cmd(f'tc -s qdisc show dev {station.wintfs[0].name}')
        match = re.search(r'backlog (\d+)b', output)
        if match:
            return int(match.group(1)) / 1000
    except Exception as e:
        print(f"Queue length error: {e}")
    return 0


def select_next_hop(agent, src, neighbors, previous_latency, destination):
    avg_signal_strength = get_average_signal_strength(src, neighbors)
    queue_length = get_queue_length(src)
    state = np.array([src.position[0], src.position[1], avg_signal_strength, queue_length])
    return qfanet_dqn_routing(agent, state, src, neighbors, previous_latency, destination, latency_measure_func)

def topology():
    net = Mininet_wifi(controller=Controller, link=wmediumd, wmediumd_mode=interference)

    stations = [net.addStation(f'sta{i+1}', position=f"{random.randint(10, 490)},{random.randint(10, 490)},0") for i in range(25)]

    # ap1 = net.addAccessPoint('ap1', ssid='ssid', mode='g', channel='1', position='150,150,0', range=100)
    # ap2 = net.addAccessPoint('ap2', ssid='ssid', mode='g', channel='6', position='350,150,0', range=100)
    # ap3 = net.addAccessPoint('ap3', ssid='ssid', mode='g', channel='11', position='250,350,0', range=100)

    h1 = net.addHost('h1', ip='10.0.0.1/8')
    h2 = net.addHost('h2', ip='10.0.0.2/8')
    h3 = net.addHost('h3', ip='10.0.0.3/8')

    # net.addLink(h1, ap1)
    # net.addLink(h2, ap2)
    # net.addLink(h3, ap3)

    for i, station in enumerate(stations):
        if i < num_stations_ap1:
            net.addLink(station, h1)
        elif i < num_stations_ap1 + num_stations_ap2:
            net.addLink(station, h2)
        else:
            net.addLink(station, h3)

    net.configureWifiNodes()
    PropagationModel(model="logDistance", exp=4)

    c0 = net.addController('c0')
    net.build()
    c0.start()
    # ap1.start([c0])
    # ap2.start([c0])
    # ap3.start([c0])

    mobility(stations, model='RandomDirection', max_x=500, max_y=500, min_v=0.5, max_v=1.0)

    agent = QFANET_DQN_Agent(state_dim=4, action_dim=10)

    exp_thread = threading.Thread(target=experiment, args=(agent, stations, h1))
    exp_thread.start()

    CLI(net)
    net.stop()

def experiment(agent, stations, destination):
    timestamps, delays, jitters, throughputs = [], [], [], []
    last_latency = None

    # Example: divide stations into 5 clusters of 5 stations each
    clusters = [stations[i:i+5] for i in range(0, len(stations), 5)]

    for cluster in clusters:
        cluster_head = select_cluster_head(cluster)

        for sta in cluster:
            if sta == cluster_head:
                continue  # Skip sending from cluster head to itself

            neighbors = [n for n in cluster if n != sta]
            prev_latency = last_latency if last_latency is not None else 0
            next_hop = select_next_hop(agent, sta, neighbors, prev_latency, destination)

            start_time = time.time()
            sta.cmd(f'echo "X" | nc -u -w1 {next_hop.IP()} 5001')
            end_time = time.time()

            latency = (end_time - start_time) * 1000
            timestamps.append(end_time)
            delays.append(latency)

            jitter = abs(latency - last_latency) if last_latency is not None else 0
            jitters.append(jitter)
            last_latency = latency

            throughput = (64 * 8) / (latency / 1000 + 0.0001) / 1000
            throughputs.append(throughput)

            print(f"{sta.name} -> {next_hop.name} -> {destination.name} : Delay={latency:.2f} ms, Jitter={jitter:.2f} ms, Throughput={throughput:.2f} kbps")

    plot_results(timestamps, delays, jitters, throughputs)

def plot_results(timestamps, delays, jitters, throughputs):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, delays, marker='o', label='Delay (ms)')
    plt.ylabel('Delay (ms)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, jitters, marker='x', color='orange', label='Jitter (ms)')
    plt.ylabel('Jitter (ms)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, throughputs, marker='s', color='green', label='Throughput (kbps)')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (kbps)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    topology()