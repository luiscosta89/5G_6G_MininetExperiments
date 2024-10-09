from pox.core import core
import pox.openflow.libopenflow_01 as of
import time
import numpy as np

log = core.getLogger()

# Import the Q-FANET class (same as defined earlier)
from qfanet import QFANET

class QFANETController(object):
    def __init__(self):
        self.qfanet = None
        self.uavs = {}
        self.num_uavs = 5
        self.update_frequency = 0.1  # 100 ms update frequency
        self.expiration_time = 0.3  # 300 ms expiration time
        self.controller_start()

    def controller_start(self):
        # Initialize the Q-FANET algorithm
        self.qfanet = QFANET(num_nodes=self.num_uavs)
        log.info("Q-FANET Controller initialized.")
        # Start periodic neighbor discovery
        self.periodic_neighbor_discovery()

    def _handle_ConnectionUp(self, event):
        log.info("Switch %s has connected" % event.connection)
        # Handle new connections from switches (APs in our case)

    def _handle_PacketIn(self, event):
        # Handle incoming packets (e.g., HELLO messages from UAVs)
        packet = event.parsed
        src = str(packet.src)  # Source node ID (UAV)
        dst = str(packet.dst)  # Destination node ID (UAV or host)

        if packet.type == 0x88cc:  # Example: Custom HELLO packet type
            self.handle_hello_packet(src, packet)

        # Install flow rules for the calculated route if necessary
        self.install_flow_rules(event.connection, src, dst)

    def handle_hello_packet(self, src, packet):
        # Handle HELLO packet and update neighbor tables in Q-FANET
        node_idx = int(src[-1]) - 1  # Assuming UAV names are sta1, sta2, etc.
        hello_info = self.parse_hello_packet(packet)
        self.qfanet.receive_hello_packet(node_idx, hello_info)

    def parse_hello_packet(self, packet):
        # Extract HELLO packet information
        return {
            'node': packet.src,
            'position': np.random.uniform(0, 100),  # Example: Simulated geographic location
            'energy': np.random.uniform(50, 100),  # Example: Simulated energy level
            'queue_delay': np.random.uniform(0.01, 0.1),  # Simulated queuing delay
            'q_value': np.random.random()  # Example: Random Q-value for demonstration
        }

    def periodic_neighbor_discovery(self):
        # Perform periodic neighbor discovery and update Q-FANET's tables
        log.info("Performing periodic neighbor discovery...")
        while True:
            self.qfanet.neighbor_discovery()

            # Simulate selecting the best route for each UAV
            for uav_idx in range(self.num_uavs):
                destination_idx = np.random.choice(range(self.num_uavs))
                if uav_idx != destination_idx:
                    best_route = self.qfanet.get_best_route(uav_idx, destination_idx)
                    log.info(f"Best route from UAV {uav_idx + 1} to UAV {destination_idx + 1}: {best_route}")

            time.sleep(self.update_frequency)

    def install_flow_rules(self, connection, src, dst):
        # Install flow rules in the OpenFlow switches based on the Q-FANET decision
        log.info(f"Installing flow from {src} to {dst}")
        msg = of.ofp_flow_mod()
        msg.match = of.ofp_match()
        msg.match.dl_src = src
        msg.match.dl_dst = dst
        msg.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        connection.send(msg)

def launch():
    core.registerNew(QFANETController)
