#!/usr/bin/env python3
"""
Model Accuracy Benchmark
------------------------
Tests different model configurations using Leave-One-Out-Location (LOLO)
cross-validation on the existing fingerprint data.

For each unique location, we:
  1. Remove all records from that location (test set)
  2. Train on the remaining records
  3. Predict each test record and measure error

This gives us a realistic estimate of accuracy at unseen positions.
"""

import json
import math
import pickle
import numpy as np
from collections import defaultdict

# ── Load data ──
def load_records(path="fingerprints_imu.json"):
    with open(path, "r") as f:
        return json.load(f)

def build_map_from_records(records, stability_threshold=0.6, min_stable_aps=3):
    """Build probabilistic map from a list of records."""
    loc_scans = defaultdict(list)
    for r in records:
        key = (round(float(r["x"]), 2), round(float(r["y"]), 2))
        loc_scans[key].append((r.get("label", ""), r.get("networks", {})))

    map_data = {}
    for (x, y), scans in loc_scans.items():
        n_scans = len(scans)
        labels = [lbl for lbl, _ in scans if lbl]
        label = max(set(labels), key=labels.count) if labels else f"pt_{x}_{y}"
        
        bssid_rssi = defaultdict(list)
        for _, nets in scans:
            for bssid, rssi in nets.items():
                bssid_rssi[bssid].append(float(rssi))
        
        stable_aps = {}
        for bssid, rssi_list in bssid_rssi.items():
            if len(rssi_list) / n_scans >= stability_threshold:
                mu = float(np.mean(rssi_list))
                sigma = float(np.std(rssi_list))
                if sigma < 2.0:
                    sigma = 2.0
                stable_aps[bssid] = {"mu": mu, "sigma": sigma}
        
        if len(stable_aps) >= min_stable_aps:
            map_data[(x, y)] = {"label": label, "n_scans": n_scans, "aps": stable_aps}
    
    return map_data

def cosine_knn_predict(p_map, scan, k=5, default_rssi=-100):
    """Predict location using Cosine KNN."""
    if not scan or not p_map:
        return None, None
    
    locations = list(p_map.keys())
    all_bssids = set()
    for loc_data in p_map.values():
        all_bssids.update(loc_data["aps"].keys())
    bssid_list = sorted(all_bssids)
    bssid_idx = {b: i for i, b in enumerate(bssid_list)}
    n_feats = len(bssid_list)
    
    # Build reference vectors
    ref_vectors = np.zeros((len(locations), n_feats), dtype=np.float64)
    for i, loc in enumerate(locations):
        vec = np.full(n_feats, default_rssi, dtype=np.float64)
        for bssid, stats in p_map[loc]["aps"].items():
            if bssid in bssid_idx:
                vec[bssid_idx[bssid]] = stats["mu"]
        ref_vectors[i] = vec - default_rssi
    
    # Build test vector
    test_vec = np.full(n_feats, default_rssi, dtype=np.float64)
    for bssid, rssi in scan.items():
        if bssid in bssid_idx:
            test_vec[bssid_idx[bssid]] = float(rssi)
    test_shifted = test_vec - default_rssi
    tn = np.linalg.norm(test_shifted)
    if tn == 0:
        return None, None
    test_unit = test_shifted / tn
    
    ref_norms = np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    ref_norms[ref_norms == 0] = 1.0
    ref_unit = ref_vectors / ref_norms
    
    cos_sims = ref_unit @ test_unit
    
    k_actual = min(k, len(locations))
    topk_idx = np.argpartition(-cos_sims, k_actual)[:k_actual]
    topk_sims = cos_sims[topk_idx]
    
    weights = np.maximum(topk_sims, 1e-6)
    total_w = weights.sum()
    
    pred_x = sum(weights[j] * locations[topk_idx[j]][0] for j in range(k_actual)) / total_w
    pred_y = sum(weights[j] * locations[topk_idx[j]][1] for j in range(k_actual)) / total_w
    
    return float(pred_x), float(pred_y)


def gaussian_predict(p_map, scan, top_n=5):
    """Predict location using Gaussian/Probabilistic matching."""
    if not scan or not p_map:
        return None, None
    
    locations = list(p_map.keys())
    log_likelihoods = []
    
    for loc in locations:
        aps = p_map[loc]["aps"]
        ll = 0
        matched = 0
        for bssid, rssi in scan.items():
            if bssid in aps:
                mu = aps[bssid]["mu"]
                sigma = aps[bssid]["sigma"]
                # Gaussian log-likelihood
                diff = float(rssi) - mu
                ll -= (diff ** 2) / (2 * sigma ** 2)
                matched += 1
        
        if matched < 3:
            ll = -1e9  # Not enough matching APs
        
        log_likelihoods.append(ll)
    
    log_likelihoods = np.array(log_likelihoods)
    
    # Get top N
    top_n_actual = min(top_n, len(locations))
    topk_idx = np.argpartition(-log_likelihoods, top_n_actual)[:top_n_actual]
    topk_ll = log_likelihoods[topk_idx]
    
    # Convert to weights (softmax-like)
    topk_ll -= topk_ll.max()
    weights = np.exp(topk_ll)
    total_w = weights.sum()
    if total_w == 0:
        return None, None
    
    pred_x = sum(weights[j] * locations[topk_idx[j]][0] for j in range(top_n_actual)) / total_w
    pred_y = sum(weights[j] * locations[topk_idx[j]][1] for j in range(top_n_actual)) / total_w
    
    return float(pred_x), float(pred_y)


def hybrid_predict(p_map, scan, k=5, cosine_weight=0.5):
    """Combine Cosine KNN and Gaussian predictions."""
    cx, cy = cosine_knn_predict(p_map, scan, k=k)
    gx, gy = gaussian_predict(p_map, scan)
    
    if cx is None and gx is None:
        return None, None
    if cx is None:
        return gx, gy
    if gx is None:
        return cx, cy
    
    w1 = cosine_weight
    w2 = 1 - cosine_weight
    return w1 * cx + w2 * gx, w1 * cy + w2 * gy


def evaluate_lolo(records, predict_fn, stability_threshold=0.6, min_stable_aps=3):
    """Leave-One-Out-Location cross-validation."""
    # Group records by location
    loc_groups = defaultdict(list)
    for r in records:
        key = (round(float(r["x"]), 2), round(float(r["y"]), 2))
        loc_groups[key].append(r)
    
    errors = []
    locations_tested = 0
    
    for test_loc, test_records in loc_groups.items():
        # Train on everything except this location
        train_records = [r for r in records if (round(float(r["x"]), 2), round(float(r["y"]), 2)) != test_loc]
        
        if len(train_records) < 10:
            continue
        
        p_map = build_map_from_records(train_records, stability_threshold, min_stable_aps)
        
        if not p_map:
            continue
        
        locations_tested += 1
        
        for test_r in test_records:
            scan = test_r.get("networks", {})
            if not scan:
                continue
            
            px, py = predict_fn(p_map, scan)
            if px is None:
                continue
            
            actual_x, actual_y = float(test_r["x"]), float(test_r["y"])
            error = math.sqrt((px - actual_x) ** 2 + (py - actual_y) ** 2)
            errors.append(error)
    
    if not errors:
        return {"mean": float("inf"), "median": float("inf"), "within_5m": 0, "n": 0}
    
    errors = np.array(errors)
    return {
        "mean": round(float(np.mean(errors)), 2),
        "median": round(float(np.median(errors)), 2),
        "within_5m": round(float(np.mean(errors <= 5.0) * 100), 1),
        "within_3m": round(float(np.mean(errors <= 3.0) * 100), 1),
        "max": round(float(np.max(errors)), 2),
        "n": len(errors),
        "locs_tested": locations_tested,
    }


# ── Main benchmark ──
if __name__ == "__main__":
    print("Loading fingerprint data...")
    records = load_records("fingerprints_imu.json")
    print(f"Loaded {len(records)} records\n")
    
    print("=" * 70)
    print("BENCHMARK: Leave-One-Out-Location Cross-Validation")
    print("=" * 70)
    
    configs = [
        # (Name, predict function, stability_threshold, min_stable_aps)
        ("Cosine KNN K=1",         lambda pm, s: cosine_knn_predict(pm, s, k=1),  0.6, 3),
        ("Cosine KNN K=2",         lambda pm, s: cosine_knn_predict(pm, s, k=2),  0.6, 3),
        ("Cosine KNN K=3",         lambda pm, s: cosine_knn_predict(pm, s, k=3),  0.6, 3),
        ("Cosine KNN K=4",         lambda pm, s: cosine_knn_predict(pm, s, k=4),  0.6, 3),
        ("Cosine KNN K=5 (current)", lambda pm, s: cosine_knn_predict(pm, s, k=5),  0.6, 3),
        ("Cosine KNN K=6",         lambda pm, s: cosine_knn_predict(pm, s, k=6),  0.6, 3),
        ("Cosine KNN K=7",         lambda pm, s: cosine_knn_predict(pm, s, k=7),  0.6, 3),
        ("Cosine K=5 Stab=0.2",     lambda pm, s: cosine_knn_predict(pm, s, k=5),  0.2, 3),
        ("Cosine K=5 Stab=0.3",     lambda pm, s: cosine_knn_predict(pm, s, k=5),  0.3, 3),
        ("Cosine K=5 Stab=0.4",     lambda pm, s: cosine_knn_predict(pm, s, k=5),  0.4, 3),
        ("Cosine K=5 Stab=0.5",     lambda pm, s: cosine_knn_predict(pm, s, k=5),  0.5, 3),
        ("Cosine K=3 Stab=0.3",     lambda pm, s: cosine_knn_predict(pm, s, k=3),  0.3, 3),
        ("Cosine K=3 Stab=0.4",     lambda pm, s: cosine_knn_predict(pm, s, k=3),  0.4, 3),
        ("Cosine K=4 Stab=0.4",     lambda pm, s: cosine_knn_predict(pm, s, k=4),  0.4, 3),
    ]
    
    results = []
    for name, pred_fn, stab, min_ap in configs:
        print(f"\nTesting: {name} ...")
        r = evaluate_lolo(records, pred_fn, stability_threshold=stab, min_stable_aps=min_ap)
        results.append((name, r))
        print(f"  Mean: {r['mean']}m | Median: {r['median']}m | ≤3m: {r['within_3m']}% | ≤5m: {r['within_5m']}% | Max: {r['max']}m | N={r['n']}")
    
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Mean':>6} {'Median':>7} {'≤3m':>6} {'≤5m':>6} {'Max':>6}")
    print("-" * 70)
    
    # Sort by mean error
    results.sort(key=lambda x: x[1]["mean"])
    for name, r in results:
        marker = " ★" if r["mean"] == results[0][1]["mean"] else ""
        print(f"{name:<30} {r['mean']:>5.1f}m {r['median']:>6.1f}m {r['within_3m']:>5.1f}% {r['within_5m']:>5.1f}% {r['max']:>5.1f}m{marker}")
    
    print("\n★ = Best model (lowest mean error)")
    print("=" * 70)
