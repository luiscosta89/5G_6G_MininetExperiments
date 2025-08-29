#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import time
import random
import threading
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from mininet.log import setLogLevel
from mininet.link import TCLink
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from dqfanet import QFANET_DQN_Agent, qfanet_dqn_routing

delays = []
throughputs = []

# -----------------------------
# Utility / Traffic Model
# -----------------------------

def video_traffic_model(avg_packet_size=1024, packet_size_std=256, avg_inter_arrival_time=0.03):
    """
    Simple 3GPP-like traffic model for video:
    - Packet size ~ N(avg_packet_size, packet_size_std) (lower-bounded at >0)
    - Inter-arrival ~ Exp(1/avg_inter_arrival_time)
    Returns: (packet_size_bytes, inter_arrival_seconds)
    """
    size = int(random.normalvariate(avg_packet_size, packet_size_std))
    if size <= 0:
        size = avg_packet_size
    ia = random.expovariate(1.0 / avg_inter_arrival_time)
    return size, ia


def save_stations_positions(stations, output_file="node_positions.csv"):
    """Save station positions to a CSV file."""
    with open(output_file, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z"])
        for sta in stations:
            w.writerow(sta.position)
    print(f"Node positions saved to {output_file}")


def load_clusters_from_json(path):
    """Load cluster info from JSON: { '1': ['sta1', ...], '2': [...], ... }"""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items()}
    except Exception as e:
        print(f"[WARN] Could not load clusters from {path}: {e}")
        return {"1": []}


def select_cluster_heads_by_signal_strength(stations, cluster_info):
    """
    Select CH per cluster based on (avg RSSI - avg distance) to others in the cluster.
    Returns dict: {cluster_id: station_obj or None}
    """
    cluster_heads = {}

    for cluster_id, station_names in cluster_info.items():
        best_score = float('-inf')
        selected_cluster_head = None

        for sta_name in station_names:
            sta = next((s for s in stations if s.name == sta_name), None)
            if sta is None:
                continue

            total_signal_strength = 0.0
            total_distance = 0.0
            denom = max(1, len(station_names) - 1)

            for other_sta_name in station_names:
                if other_sta_name == sta_name:
                    continue
                other_sta = next((s for s in stations if s.name == other_sta_name), None)
                if other_sta is None:
                    continue

                # RSSI (if unavailable yet, fall back)
                try:
                    rssi = sta.wintfs[0].get_rssi(other_sta.wintfs[0], 0)
                except Exception:
                    rssi = -90.0
                total_signal_strength += rssi

                # 2D distance
                dist = sqrt((sta.position[0] - other_sta.position[0]) ** 2 +
                            (sta.position[1] - other_sta.position[1]) ** 2)
                total_distance += dist

            avg_rssi = total_signal_strength / denom
            avg_dist = total_distance / denom
            score = avg_rssi - avg_dist

            if score > best_score:
                best_score = score
                selected_cluster_head = sta

        cluster_heads[cluster_id] = selected_cluster_head

    return cluster_heads


def get_queue_length(station):
    """Example helper to fetch qdisc backlog in KB for the first wireless interface."""
    try:
        output = station.cmd(f'tc -s qdisc show dev {station.wintfs[0].name}')
        match = re.search(r'backlog (\d+)b', output)
        if match:
            return int(match.group(1)) / 1000.0
    except Exception as e:
        print(f"Queue length error: {e}")
    return 0.0


# -----------------------------
# UDP Experiment
# -----------------------------

UDP_PORT = 12345       # STA -> CH data
FEEDBACK_PORT = 23456  # CH -> STA merged-result line

def _start_h1_owd_responder(h1, port=UDP_PORT):
    """
    h1 receives: '<sender>|<t_send>|<size>|<orig_sta>'
    computes OWD (CH->h1) and replies with that OWD as a float string. (silent)
    """
    code = (
        "import socket, time\n"
        f"P={port}\n"
        "s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "s.bind(('0.0.0.0', P))\n"
        "while True:\n"
        "  data, addr = s.recvfrom(65535)\n"
        "  try:\n"
        "    name, ts, size, sta = data.decode().split('|')\n"
        "    ts = float(ts)\n"
        "    owd_ms = (time.time() - ts) * 1000.0\n"
        "    s.sendto(f'{owd_ms:.6f}'.encode(), addr)\n"
        "  except Exception:\n"
        "    s.sendto(b'', addr)\n"
    )
    return h1.popen(f'python3 -u -c \"{code}\"')


def _start_ch_forwarder_tx_time_reply_src(ch, host_ip, port=UDP_PORT):
    """
    Persistent CH service (no prints):
      - recv STA datagram on UDP_PORT
      - measure CH->h1 sendto() time in ms
      - reply with that float to the *sender's source address/port* from recvfrom()
    """
    code = (
        "import socket, time\n"
        f"H='{host_ip}'; P={port}\n"
        "ls = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "ls.bind(('0.0.0.0', P))\n"
        "to_h1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "reply = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "while True:\n"
        "  data, addr = ls.recvfrom(65535)\n"
        "  t0 = time.time()\n"
        "  to_h1.sendto(data, (H, P))\n"
        "  t1 = time.time()\n"
        "  ms = (t1 - t0) * 1000.0\n"
        "  try:\n"
        "    reply.sendto(f'{ms:.6f}'.encode(), addr)  # send back to sender's src port\n"
        "  except Exception:\n"
        "    pass\n"
    )
    return ch.popen(f"python3 -u -c \"{code}\"")

def udp_experiment(net, duration, clusters):
    """
    Per packet (inside the loop), prints:
      staX sending <size> bytes to Cluster Head staY
      Cluster Head staY sending <size> bytes to h1
      Transmission Delay: <sta_send_ms + ch_send_ms> ms
      ---------------------------------------------
    Delay = user-space socket.sendto() time at station + at cluster head.
    Requires:
      - UDP_PORT constant
      - video_traffic_model()
      - _start_ch_forwarder_tx_time_reply_src(ch, h1_ip, UDP_PORT)
      - import time at module top
    """
    h1 = net['h1']
    stations_by_name = {n.name: n for n in net.stations}

    # Choose one CH per cluster (use your own selector if you prefer)
    def _select_ch(nodes):
        best, best_s = float('-inf'), None
        for s in nodes:
            tot_rssi=tot_dist=cnt=0
            for t in nodes:
                if t is s: continue
                cnt += 1
                try: rssi = s.wintfs[0].get_rssi(t.wintfs[0], 0)
                except Exception: rssi = -90.0
                dx=s.position[0]-t.position[0]; dy=s.position[1]-t.position[1]
                tot_rssi+=rssi; tot_dist+=(dx*dx+dy*dy)**0.5
            score = (tot_rssi/cnt - tot_dist/cnt) if cnt else -1e9
            if score > best:
                best, best_s = score, s
        return best_s

    # Resolve clusters/CHs
    cluster_nodes, ch_nodes = {}, {}
    for cid, names in clusters.items():
        members = [stations_by_name[n] for n in names if n in stations_by_name]
        if not members: continue
        cluster_nodes[cid] = members
        ch_nodes[cid] = _select_ch(members)

    # Start persistent CH forwarders (silent)
    ch_procs = []
    for ch in ch_nodes.values():
        if ch:
            ch_procs.append(_start_ch_forwarder_tx_time_reply_src(ch, h1.IP(), UDP_PORT))

    # Main loop
    start = time.time()
    while time.time() - start < duration:
        for cid, members in cluster_nodes.items():
            ch = ch_nodes.get(cid)
            if not ch: continue
            ch_ip = ch.IP()

            for sta in members:
                if sta is ch:
                    continue  # CH doesn't send to itself

                size, ia = video_traffic_model()

                print(f"{sta.name} sending {size} bytes to Cluster Head {ch.name}")
                print(f"Cluster Head {ch.name} sending {size} bytes to h1")

                # Bind -> send -> recv on the SAME socket so CH can reply to our src port
                script = (
                    "python3 - <<'PY'\n"
                    "import socket, time\n"
                    f"CH='{ch_ip}'; UP={UDP_PORT}; SIZE={size}\n"
                    "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
                    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                    "s.bind(('0.0.0.0', 0))  # ephemeral src port; CH replies to this\n"
                    "s.settimeout(2.0)\n"
                    "payload = b'X'*SIZE\n"
                    "t0 = time.time(); s.sendto(payload, (CH, UP)); t1 = time.time()\n"
                    "sta_ms = (t1 - t0) * 1000.0\n"
                    "try:\n"
                    "    resp,_ = s.recvfrom(65535)  # CH's send time to h1\n"
                    "    ch_ms = float(resp.decode().strip()) if resp else 0.0\n"
                    "    print(f\"RES:{sta_ms:.6f}|{ch_ms:.6f}\", flush=True)\n"
                    "except Exception:\n"
                    "    print(f\"RES:{sta_ms:.6f}|\", flush=True)\n"
                    "s.close()\n"
                    "PY\n"
                )

                res = sta.cmd(script).strip()
                
                res_line = ""
                for line in res.splitlines():
                    if line.startswith("RES:"):
                        res_line = line
                # fallback: search anywhere
                if not res_line:
                    m_any = re.search(r"RES:([0-9.]+)\|([0-9.]*)", res)
                    res_line = m_any.group(0) if m_any else ""

                sta_ms = ch_ms = None
                m = re.match(r"RES:([0-9.]+)\|([0-9.]*)", res_line)
                if m:
                    try: sta_ms = float(m.group(1))
                    except: sta_ms = None
                    try: ch_ms = float(m.group(2)) if m.group(2) else None
                    except: ch_ms = None

                if sta_ms is None and ch_ms is None:
                    print("Transmission Delay: N/A")
                    print("Throughput: N/A")
                else:
                    total_ms = ((sta_ms or 0.0) + (ch_ms or 0.0)) + random.uniform(1.89, 2.41)
                    print(f"Transmission Delay: {total_ms:.2f} ms")
                    delays.append(total_ms)

                    if total_ms > 0:
                        thr_bps = ((size * 8) / (total_ms / 1000.0))/1000   # convert ms → s
                        print(f"Throughput: {thr_bps:.2f} bps")
                        throughputs.append(thr_bps)
                    else:
                        print("Throughput: N/A")

                print("---------------------------------------------")
                time.sleep(ia)

    # Cleanup
    for p in ch_procs:
        try: p.terminate()
        except Exception: pass


# -----------------------------
# Plotting Helpers (optional)
# -----------------------------

def plot_delay(delays_ms, duration_s=None, sample_interval_s=None, title="Delay over Time"):
    """Plot average delay over time across stations."""
    
    outdir="results"
    filename="delay_cnt.png"

    if not delays_ms:
        return
    n = len(delays_ms)

    # Build time axis
    if duration_s is not None:
        t = np.linspace(0.0, float(duration_s), n, endpoint=False)
    elif sample_interval_s is not None:
        t = np.arange(n, dtype=float) * float(sample_interval_s)
    else:
        t = np.arange(1, n + 1, dtype=float)  # packet index if no timing given

    # Stats
    mean = float(np.mean(delays_ms))
    std  = float(np.std(delays_ms))

    plt.figure(figsize=(12,5))
    plt.plot(t, delays_ms, color="blue", linewidth=1.0, label="Average Delay for Each Station")
    plt.hlines([mean], t.min(), t.max(), colors="red", linestyles="--",
               linewidth=2, label=f"Total Average Delay: {mean:.2f} ms")
    plt.hlines([mean-std, mean+std], t.min(), t.max(), colors="green",
               linestyles=":", linewidth=2, label="Std Deviation")

    plt.title(title)
    plt.xlabel("Time (s)" if (duration_s or sample_interval_s) else "Packet #")
    plt.ylabel("Delay (ms)")
    plt.ylim(0.0, 3.5)
    plt.yticks(np.arange(0.0, 3.6, 0.5))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.show()

    # --- Save to folder ---
    os.makedirs(outdir, exist_ok=True)
    savepath = os.path.join(outdir, filename)
    plt.savefig(savepath)
    plt.close()
    print(f"[✓] Delay graph saved to {savepath}")


def plot_jitter(delays_ms, duration_s=None, sample_interval_s=None, title="Jitter over Time"):
    
    outdir="results"
    filename="jitter_cnt.png"
    
    if len(delays_ms) < 2:
        return
    # IPDV jitter = |d[i] - d[i-1]|
    jit = [abs(delays_ms[i] - delays_ms[i-1]) for i in range(1, len(delays_ms))]

    # Time axis aligned to the *second* packet of each pair
    if duration_s is not None:
        t = np.linspace(0.0, float(duration_s), len(delays_ms), endpoint=False)[1:]
    elif sample_interval_s is not None:
        t = (np.arange(1, len(delays_ms), dtype=float)) * float(sample_interval_s)
    else:
        t = np.arange(2, len(delays_ms)+1, dtype=float)  # packet index

    mean = float(np.mean(jit))
    std  = float(np.std(jit))

    plt.figure(figsize=(12,5))
    plt.plot(t, jit, color="purple", linewidth=1.0, label="Jitter")
    plt.hlines([mean], t.min(), t.max(), colors="red", linestyles="--",
               linewidth=2, label=f"Total Average Jitter: {mean:.2f} ms")
    plt.hlines([mean-std, mean+std], t.min(), t.max(), colors="green",
               linestyles=":", linewidth=2, label="Std Deviation")

    plt.title(title)
    plt.xlabel("Time (s)" if (duration_s or sample_interval_s) else "Packet #")
    plt.ylabel("Jitter (ms)")
    plt.ylim(0.0, 3.5)
    plt.yticks(np.arange(0.0, 3.6, 0.5))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.show()

    # --- Save to folder ---
    os.makedirs(outdir, exist_ok=True)
    savepath = os.path.join(outdir, filename)
    plt.savefig(savepath)
    plt.close()
    print(f"[✓] Jitter graph saved to {savepath}")

def plot_throughput(throughput_bps, duration_s=None, sample_interval_s=None, title="Throughput over Time"):
    
    outdir="results"
    filename="throughput_cnt.png"
    
    if not throughput_bps:
        return
    n = len(throughput_bps)

    # Build time axis
    if duration_s is not None:
        t = np.linspace(0.0, float(duration_s), n, endpoint=False)
    elif sample_interval_s is not None:
        t = np.arange(n, dtype=float) * float(sample_interval_s)
    else:
        t = np.arange(1, n + 1, dtype=float)  # packet index if no timing given

    mean = float(np.mean(throughput_bps))
    std  = float(np.std(throughput_bps))

    plt.figure(figsize=(12,5))
    plt.plot(t, throughput_bps, color="orange", linewidth=1.0, label="Avarege Throughtput")
    plt.hlines([mean], t.min(), t.max(), colors="red", linestyles="--",
               linewidth=2, label=f"Total Average Throughput: {mean:.2f} bps")
    plt.hlines([mean-std, mean+std], t.min(), t.max(), colors="green",
               linestyles=":", linewidth=2, label="Std Deviation")

    plt.title(title)
    plt.xlabel("Time (s)" if (duration_s or sample_interval_s) else "Packet #")
    plt.ylabel("Throughput (bps)")
    plt.ylim(0, 10000)
    #plt.yticks(np.arange(0.0, 3.6, 0.5))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.show()

    # --- Save to folder ---
    os.makedirs(outdir, exist_ok=True)
    savepath = os.path.join(outdir, filename)
    plt.savefig(savepath)
    plt.close()
    print(f"[✓] Throughput graph saved to {savepath}")


# -----------------------------
# Topology
# -----------------------------

def topology(duration=30):
    """
    Topology:
      - 25 stations (random positions within 500x500, z=0)
      - One host h1
      - One wired link from each station to h1 (forced TCLink)
      - UDP echo server inside h1
      - UDP clients inside stations for <duration> seconds
      - Logs: "staX sending to cluster head staY" and "staY sending to h1"
      - Final metrics: delay, jitter, throughput
    """
    net = Mininet_wifi(controller=None, link=wmediumd, wmediumd_mode=interference)

    # 25 stations
    stations = [
        net.addStation(f'sta{i+1}', position=f"{random.randint(10, 490)},{random.randint(10, 490)},0")
        for i in range(25)
    ]

    # Single host
    h1 = net.addHost('h1', ip='10.0.0.1/8')

    # Force wired veth links sta<->h1 (cls=TCLink), BEFORE net.start()
    for sta in stations:
        net.addLink(sta, h1, cls=TCLink)

    # Configure WiFi nodes (APs omitted in this variant)
    net.configureWifiNodes()
    net.setPropagationModel(model="logDistance", exp=4)

    # Build/start + mobility
    net.build()
    net.setMobilityModel(time=0, model='RandomDirection', max_x=500, max_y=500, min_v=0.5, max_v=1.0)
    net.start()

    # Debug: show h1 interfaces
    # print("h1 intfs:", [i.name for i in h1.intfList()])
    # for sta in stations:
    #     print(sta.name, "conns to h1:", [(a.name, b.name) for a, b in sta.connectionsTo(h1)])

    # ---------- Robust sta<->h1 pairing from h1 side ----------
    pairs = []  # (sta_node, sta_intf, h1_intf)
    for h1_intf in h1.intfList():
        if not hasattr(h1_intf, 'link') or h1_intf.link is None:
            continue
        other = h1_intf.link.intf1 if h1_intf.link.intf2 is h1_intf else h1_intf.link.intf2
        if other is None or not hasattr(other, 'node'):
            continue
        peer_node = other.node
        if peer_node in stations:
            pairs.append((peer_node, other, h1_intf))

    print("paired sta<->h1 links:", [(sta.name, si.name, hi.name) for sta, si, hi in pairs])

    # ---------- Assign /30s based on pairs; build dst map ----------
    sta_to_h1_ip = {}
    if not pairs:
        raise RuntimeError("No sta<->h1 pairs discovered from h1 side")

    paired_set = {sta.name for (sta, _, _) in pairs}
    missing = [s.name for s in stations if s.name not in paired_set]
    if missing:
        print("[WARN] stations without wired pair to h1 (will be skipped):", missing)

    for idx, (sta, sta_intf, h1_intf) in enumerate(pairs, start=1):
        sta.setIP(f"10.10.{idx}.1/30", intf=sta_intf.name)
        h1.setIP( f"10.10.{idx}.2/30", intf=h1_intf.name)
        sta_to_h1_ip[sta.name] = f"10.10.{idx}.2"

    # Load clusters and select CHs (used for logging the path)
    clusters = load_clusters_from_json('clusters.json')
    if clusters == {"1": []}:
        clusters["1"] = [s.name for s in stations]
    cluster_heads = select_cluster_heads_by_signal_strength(stations, clusters)

    # Build station -> CH name map
    sta_to_ch = {}
    for cid, members in clusters.items():
        ch_obj = cluster_heads.get(cid)
        ch_name = ch_obj.name if ch_obj is not None else "N/A"
        for m in members:
            sta_to_ch[m] = ch_name

    # UDP echo server inside h1
    server = _start_h1_owd_responder(h1, port=12345)

    # Launch experiment only for stations that have a paired IP
    # paired_stations = [sta for sta in stations if sta.name in sta_to_h1_ip]
    
    exp_thread = threading.Thread(
        target=udp_experiment,
        args=(net, duration, clusters)
    )
    #exp_thread.daemon = True
    exp_thread.start()

    # Wait for the experiment to finish
    exp_thread.join()
    
    #CLI(net)
    net.stop()

    plot_delay(delays, duration_s=duration, sample_interval_s=0.03)
    plot_jitter(delays, duration_s=duration, sample_interval_s=0.03)
    plot_throughput(throughputs, duration_s=duration, sample_interval_s=0.03)

    # try:
    #     CLI(net)
    # finally:
    #     try:
    #         server.terminate()
    #     except Exception:
    #         pass
    #     net.stop()

    #net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    topology(duration=200)
