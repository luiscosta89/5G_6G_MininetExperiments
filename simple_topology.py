from mininet.net import Mininet
from mininet.node import Controller, OVSKernelAP
from mininet.link import TCLink
from mininet.log import setLogLevel, info
import time
import numpy as np

# Import the Q-FANET class
from qfanet import QFANET  # Assuming QFANET class is in a module called qfanet.py

def q_fanet_experiment():
    # Initialize Mininet with the necessary components
    net = Mininet(controller=Controller, accessPoint=OVSKernelAP, link=TCLink)

    # Add a controller
    c0 = net.addController('c0')

    # Add Access Points and mobile stations (UAVs)
    ap1 = net.addAccessPoint('ap1', ssid="new-ssid", mode="g", channel="5", position="30,30,0")
    ap2 = net.addAccessPoint('ap2', ssid="new-ssid", mode="g", channel="5", position="60,60,0")

    # Add stations (UAV nodes)
    num_uavs = 5
    uavs = []
    for i in range(1, num_uavs + 1):
        uavs.append(net.addStation(f'sta{i}', ip=f"10.0.0.{i}", position=f"{10 * i},10,0"))

    # Configure WiFi nodes and start the network
    net.configureWifiNodes()
    net.build()
    c0.start()
    ap1.start([c0])
    ap2.start([c0])

    # Set up Q-FANET algorithm
    qfanet = QFANET(num_nodes=num_uavs)
    signal_threshold = 50  # Signal strength threshold for neighbor discovery
    update_interval = 0.1  # Update every 100ms
    experiment_duration = 10  # Duration of the experiment in seconds

    # Start the experiment loop
    start_time = time.time()
    while time.time() - start_time < experiment_duration:
        # Simulate HELLO packet exchange and neighbor discovery every 100ms
        qfanet.neighbor_discovery()

        # For each UAV (station), determine the best route to a randomly selected destination
        for i, uav in enumerate(uavs):
            uav_idx = i
            destination_idx = np.random.choice(range(num_uavs))
            if uav_idx != destination_idx:
                best_route = qfanet.get_best_route(uav_idx, destination_idx, qfanet.neighbor_tables)
                print(f"Best route from UAV {uav_idx + 1} to UAV {destination_idx + 1}: {best_route}")

        # Simulate packet transmission and delay calculation here
        # You can simulate packet transmission using net.ping or another method
        # Example: net.ping([uavs[best_route[0]], uavs[best_route[-1]]])

        time.sleep(update_interval)

    # Stop the network
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    q_fanet_experiment()
