#!/usr/bin/env python3
"""
Build Probabilistic WiFi Map
----------------------------
Merges the IMU fingerprint json files and computes the Mean (mu) and
Standard Deviation (sigma) for each stable BSSID at each (x, y) location.
"""

import json
import math
import pickle
import numpy as np
from collections import defaultdict

STABILITY_THRESHOLD = 0.4  # AP must appear in 40% of scans at a location
MIN_STABLE_APS = 3
DEFAULT_SIGMA = 5.0        # Default std dev if an AP is highly stable (avoid div by 0)

def load_data(files):
    all_records = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_records.extend(data)
                print(f"Loaded {len(data)} records from {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return all_records

def build_probabilistic_map(records):
    # Group by location
    loc_scans = defaultdict(list)
    for r in records:
        key = (round(float(r["x"]), 2), round(float(r["y"]), 2))
        loc_scans[key].append((r.get("label", ""), r.get("networks", {})))

    map_data = {}
    
    for (x, y), scans in loc_scans.items():
        n_scans = len(scans)
        
        # Get label
        labels = [lbl for lbl, _ in scans if lbl]
        label = max(set(labels), key=labels.count) if labels else f"pt_{x}_{y}"
        
        # Aggregate RSSI per BSSID
        bssid_rssi_lists = defaultdict(list)
        for _, nets in scans:
            for bssid, rssi in nets.items():
                bssid_rssi_lists[bssid].append(float(rssi))
                
        # Find Stable APs
        stable_aps = {}
        for bssid, rssi_list in bssid_rssi_lists.items():
            if len(rssi_list) / n_scans >= STABILITY_THRESHOLD:
                mu = float(np.mean(rssi_list))
                sigma = float(np.std(rssi_list))
                if sigma < 2.0:
                    sigma = 2.0  # Put a floor on sigma to avoid extreme likelihood spikes
                
                stable_aps[bssid] = {"mu": mu, "sigma": sigma}
                
        if len(stable_aps) >= MIN_STABLE_APS:
            map_data[(x, y)] = {
                "label": label,
                "n_scans": n_scans,
                "aps": stable_aps
            }

    print(f"Built probabilistic map for {len(map_data)} unique locations.")
    return map_data

if __name__ == "__main__":
    files = ["fingerprints_imu.json", "fingerprints_imuday2.json"]
    records = load_data(files)
    print(f"Total merged records: {len(records)}")
    
    p_map = build_probabilistic_map(records)
    
    with open("probabilistic_map.pkl", "wb") as f:
        pickle.dump(p_map, f)
    print("Saved to probabilistic_map.pkl")
