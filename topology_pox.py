from mininet.net import Mininet
from mininet.node import OVSKernelAP, Controller, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel, info

def q_fanet_experiment():
    # Set up the Mininet-WiFi network
    net = Mininet(controller=RemoteController, accessPoint=OVSKernelAP, link=TCLink)

    # Add a POX remote controller (run POX with the QFANETController)
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

    # Add Access Points and UAVs
    ap1 = net.addAccessPoint('ap1', ssid="new-ssid", mode="g", channel="5", position="30,30,0")
    ap2 = net.addAccessPoint('ap2', ssid="new-ssid", mode="g", channel="5", position="60,60,0")

    # Add UAV stations
    num_uavs = 5
    uavs = []
    for i in range(1, num_uavs + 1):
        uavs.append(net.addStation(f'sta{i}', ip=f"10.0.0.{i}", position=f"{10 * i},10,0"))

    # Configure and start the network
    net.configureWifiNodes()
    net.build()
    c0.start()
    ap1.start([c0])
    ap2.start([c0])

    # Allow the experiment to run for 10 seconds
    info("*** Running the Q-FANET experiment with POX controller\n")
    net.pingAll()

    # Stop the network
    info("*** Stopping the network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    q_fanet_experiment()
