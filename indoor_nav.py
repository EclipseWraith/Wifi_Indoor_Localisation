"""
Indoor Positioning System using Wi-Fi RSS Fingerprinting
=========================================================
Algorithms:
  1. WKNN   - Weighted K-Nearest Neighbors (recommended)
  2. KNN    - Plain K-Nearest Neighbors
  3. Bayesian Probabilistic
  4. Random Forest (scikit-learn)

Usage:
  python indoor-nav.py --algo wknn --scan '{"AA:BB:CC:DD:EE:FF": -65}'
  python indoor-nav.py --algo wknn --demo
  python indoor-nav.py --benchmark          # compare all algorithms
  python indoor-nav.py --benchmark --k 3    # test with k=3
"""

import json
import math
import argparse
import random
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# 1. LOAD & BUILD FINGERPRINT DATABASE
# ──────────────────────────────────────────────

def load_fingerprints(path="fingerprints_imu.json"):
    with open(path) as f:
        raw = json.load(f)

    # Collect all BSSIDs
    all_bssids = sorted({b for d in raw for b in d["networks"]})

    # Average RSSI per location label
    label_data = defaultdict(list)
    for d in raw:
        label_data[d["label"]].append({"x": d["x"], "y": d["y"], "networks": d["networks"]})

    fingerprint_db = {}
    for label, items in label_data.items():
        bssid_rssi = defaultdict(list)
        for item in items:
            for b, rssi in item["networks"].items():
                bssid_rssi[b].append(rssi)
        fingerprint_db[label] = {
            "x":    sum(i["x"] for i in items) / len(items),
            "y":    sum(i["y"] for i in items) / len(items),
            "rssi": {b: sum(v) / len(v) for b, v in bssid_rssi.items()},
            "count": len(items),
        }

    print(f"[DB] Loaded {len(raw)} samples · {len(fingerprint_db)} locations · {len(all_bssids)} unique BSSIDs")
    return raw, fingerprint_db, all_bssids


# ──────────────────────────────────────────────
# 2. FEATURE VECTOR HELPER
# ──────────────────────────────────────────────

MISSING_RSSI = -100  # value used when a BSSID is not visible

def build_feature_vector(scan: dict, all_bssids: list) -> np.ndarray:
    """Convert a {bssid: rssi} scan dict into a fixed-length numpy array."""
    return np.array([scan.get(b, MISSING_RSSI) for b in all_bssids], dtype=float)

def euclidean_distance(scan: dict, fp_rssi: dict) -> float:
    """RSS Euclidean distance between a scan and a fingerprint."""
    all_keys = set(scan) | set(fp_rssi)
    return math.sqrt(sum(
        (scan.get(b, MISSING_RSSI) - fp_rssi.get(b, MISSING_RSSI)) ** 2
        for b in all_keys
    ))


# ──────────────────────────────────────────────
# 3. ALGORITHM IMPLEMENTATIONS
# ──────────────────────────────────────────────

# ── 3a. WKNN ──────────────────────────────────
def wknn(scan: dict, db: dict, k: int = 5) -> dict:
    """
    Weighted KNN: position = weighted average of k nearest fingerprints,
    where weight = 1 / distance (closer fingerprints dominate).
    """
    dists = [
        {"label": label, "x": info["x"], "y": info["y"],
         "dist": euclidean_distance(scan, info["rssi"])}
        for label, info in db.items()
    ]
    dists.sort(key=lambda d: d["dist"])
    knn = dists[:k]

    # Inverse-distance weights (small epsilon avoids div-by-zero)
    min_d = knn[0]["dist"]
    # Added max(..., 1e-4) to safely prevent division by zero or negative weights
    weights = [1.0 / max((d["dist"] - min_d * 0.9), 1e-4) for d in knn]
    total_w = sum(weights)

    est_x = sum(d["x"] * w for d, w in zip(knn, weights)) / total_w
    est_y = sum(d["y"] * w for d, w in zip(knn, weights)) / total_w

    # Accuracy estimate: avg coord-space spread of k neighbours
    accuracy = sum(
        math.sqrt((d["x"] - est_x) ** 2 + (d["y"] - est_y) ** 2)
        for d in knn
    ) / k

    return {
        "algo":     f"WKNN (k={k})",
        "label":    knn[0]["label"],
        "x":        est_x,
        "y":        est_y,
        "accuracy": accuracy,
        "top_k":    [(d["label"], round(d["dist"], 2)) for d in knn[:3]],
    }


# ── 3b. KNN ──────────────────────────────────
def knn(scan: dict, db: dict, k: int = 5) -> dict:
    """
    Plain KNN: simple unweighted average of k nearest fingerprint coordinates.
    """
    dists = sorted(
        [{"label": lbl, "x": info["x"], "y": info["y"],
          "dist": euclidean_distance(scan, info["rssi"])}
         for lbl, info in db.items()],
        key=lambda d: d["dist"]
    )
    knn_pts = dists[:k]
    est_x = sum(d["x"] for d in knn_pts) / k
    est_y = sum(d["y"] for d in knn_pts) / k

    return {
        "algo":     f"KNN (k={k})",
        "label":    knn_pts[0]["label"],
        "x":        est_x,
        "y":        est_y,
        "accuracy": 3.0,
        "top_k":    [(d["label"], round(d["dist"], 2)) for d in knn_pts[:3]],
    }


# ── 3c. BAYESIAN PROBABILISTIC ────────────────
def bayesian(scan: dict, db: dict, sigma: float = 8.0, top_k: int = 5) -> dict:
    """
    Probabilistic model: assumes RSS at each AP follows a Gaussian distribution.
    Fixed: Calculates probability over the UNION of APs to penalize missing APs.
    """
    scores = {}
    for label, info in db.items():
        log_p = 0.0
        # Evaluate the union of BSSIDs seen in the scan OR the fingerprint
        all_keys = set(scan.keys()) | set(info["rssi"].keys())
        
        for bssid in all_keys:
            obs = scan.get(bssid, MISSING_RSSI)
            mean_rssi = info["rssi"].get(bssid, MISSING_RSSI)
            
            diff = obs - mean_rssi
            # Sum the log-likelihoods, do not average them!
            log_p -= (diff * diff) / (2 * sigma * sigma)
            
        scores[label] = log_p

    top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    max_score = top[0][1]
    
    # Exponentiate to get relative weights
    weights = [math.exp(s - max_score) for _, s in top]
    total_w = sum(weights)

    est_x = sum(db[lbl]["x"] * w for (lbl, _), w in zip(top, weights)) / total_w
    est_y = sum(db[lbl]["y"] * w for (lbl, _), w in zip(top, weights)) / total_w

    return {
        "algo":     "Bayesian Probabilistic",
        "label":    top[0][0],
        "x":        est_x,
        "y":        est_y,
        "accuracy": 2.5,
        "top_k":    [(lbl, round(s, 2)) for lbl, s in top[:3]],
    }


# ── 3d. RANDOM FOREST ─────────────────────────
class RandomForestPositioner:
    def __init__(self, n_estimators: int = 150): # Reduced slightly for faster CV
        self.n_estimators = n_estimators
        self.clf   = RandomForestClassifier(n_estimators=n_estimators, n_jobs=1, random_state=42)
        self.reg   = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=42)
        self.le    = LabelEncoder()
        self.bssids = None
        self.trained = False

    def train(self, raw: list, all_bssids: list):
        self.bssids = all_bssids
        X = np.array([
            build_feature_vector(d["networks"], all_bssids) for d in raw
        ])
        y_labels = [d["label"] for d in raw]
        y_coords = np.array([[d["x"], d["y"]] for d in raw], dtype=float)

        y_enc = self.le.fit_transform(y_labels)
        self.clf.fit(X, y_enc)
        self.reg.fit(X, y_coords)
        self.trained = True
        print(f"[RF] Trained on {len(raw)} samples · {len(all_bssids)} features")

    def predict(self, scan: dict) -> dict:
        if not self.trained:
            raise RuntimeError("Call .train() first")
        x_vec = build_feature_vector(scan, self.bssids).reshape(1, -1)
        label_idx = self.clf.predict(x_vec)[0]
        label     = self.le.inverse_transform([label_idx])[0]
        coords    = self.reg.predict(x_vec)[0]
        proba     = self.clf.predict_proba(x_vec)[0].max()
        return {
            "algo":     "Random Forest",
            "label":    label,
            "x":        float(coords[0]),
            "y":        float(coords[1]),
            "accuracy": round((1 - proba) * 10, 2),
            "confidence": round(float(proba), 3),          # e.g. 0.873
            "confidence_pct": f"{proba * 100:.1f}%",       # e.g. "87.3%"
        }


# ──────────────────────────────────────────────
# 4. BENCHMARK (cross-validation)
# ──────────────────────────────────────────────

def benchmark(raw: list, db: dict, all_bssids: list, k_val: int, n_splits: int = 5):
    print(f"\n{'='*55}")
    print(f"  BENCHMARK — {n_splits}-fold Cross Validation (k={k_val})")
    print(f"{'='*55}")

    indices = list(range(len(raw)))
    random.seed(42) # Fixed seed for reproducible benchmarks
    random.shuffle(indices)
    fold_size = len(indices) // n_splits
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_splits)]

    results = {name: [] for name in ["WKNN", "KNN", "Bayesian", "RandomForest"]}

    for fold_idx, test_idx in enumerate(folds):
        train_idx = [i for j, f in enumerate(folds) for i in f if j != fold_idx]
        train_raw = [raw[i] for i in train_idx]
        test_raw  = [raw[i] for i in test_idx]

        # Rebuild DB from training set only
        fold_db = defaultdict(list)
        for d in train_raw:
            fold_db[d["label"]].append(d)
        train_db = {}
        for label, items in fold_db.items():
            bssid_rssi = defaultdict(list)
            for item in items:
                for b, rssi in item["networks"].items():
                    bssid_rssi[b].append(rssi)
            train_db[label] = {
                "x": sum(i["x"] for i in items) / len(items),
                "y": sum(i["y"] for i in items) / len(items),
                "rssi": {b: sum(v)/len(v) for b, v in bssid_rssi.items()},
            }

        # Train RF on this fold
        rf = RandomForestPositioner(n_estimators=100)
        rf.train(train_raw, all_bssids)

        for d in test_raw:
            scan = d["networks"]
            true_x, true_y = d["x"], d["y"]

            def coord_err(res):
                return math.sqrt((res["x"] - true_x)**2 + (res["y"] - true_y)**2)

            results["WKNN"].append(coord_err(wknn(scan, train_db, k=k_val)))
            results["KNN"].append(coord_err(knn(scan, train_db, k=k_val)))
            results["Bayesian"].append(coord_err(bayesian(scan, train_db)))
            results["RandomForest"].append(coord_err(rf.predict(scan)))

        print(f"  Fold {fold_idx+1}/{n_splits} done ({len(test_raw)} test samples)")

    print(f"\n{'─'*55}")
    print(f"  {'Algorithm':<18} {'Mean Err':>10} {'Median Err':>12} {'<2 units':>10}")
    print(f"{'─'*55}")
    for name, errs in results.items():
        arr = np.array(errs)
        mean_e   = arr.mean()
        median_e = np.median(arr)
        pct_2    = (arr < 2).mean() * 100
        print(f"  {name:<18} {mean_e:>8.3f}u  {median_e:>10.3f}u  {pct_2:>8.1f}%")
    print(f"{'─'*55}")
    print("  (u = grid units;  1 unit ≈ corridor grid cell)\n")


# ──────────────────────────────────────────────
# 5. SINGLE PREDICTION
# ──────────────────────────────────────────────

def print_result(res: dict):
    print(f"\n{'─'*45}")
    print(f"  Algorithm : {res['algo']}")
    print(f"  Location  : {res['label']}")
    print(f"  Coords    : ({res['x']:.2f}, {res['y']:.2f})")
    print(f"  Accuracy  : ±{res.get('accuracy', '?'):.2f} units")
    if "confidence" in res:
        print(f"  Confidence: {res['confidence']}")
    if "top_k" in res:
        print(f"  Top-3 matches:")
        for lbl, score in res["top_k"]:
            print(f"    • {lbl}  [{score}]")
    print(f"{'─'*45}\n")


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indoor Positioning System")
    parser.add_argument("--data",      default="fingerprints_imu.json", help="Path to fingerprint JSON")
    parser.add_argument("--algo",      choices=["wknn","knn","bayesian","rf","all"], default="wknn")
    parser.add_argument("--scan",      type=str, help='JSON scan: \'{"AA:BB:CC": -65}\'')
    parser.add_argument("--demo",      action="store_true", help="Use a random sample from the dataset as test scan")
    parser.add_argument("--benchmark", action="store_true", help="Run 5-fold CV benchmark for all algorithms")
    parser.add_argument("--k",         type=int, default=5, help="K for KNN/WKNN (default: 5)")
    args = parser.parse_args()

    # Load data
    raw, db, all_bssids = load_fingerprints(args.data)

    # ── Benchmark mode ──
    if args.benchmark:
        benchmark(raw, db, all_bssids, k_val=args.k)
        return

    # ── Build scan dict ──
    if args.scan:
        scan = json.loads(args.scan)
        print(f"\n[Scan] {len(scan)} APs visible")
    elif args.demo:
        sample = random.choice(raw)
        scan   = sample["networks"]
        print(f"\n[Demo] Ground truth: '{sample['label']}' at ({sample['x']}, {sample['y']})")
        print(f"[Demo] {len(scan)} APs in scan")
    else:
        parser.print_help()
        print("\n⚠  Provide --scan or --demo or --benchmark\n")
        return

    # ── Run algorithm(s) ──
    algos_to_run = ["wknn","knn","bayesian","rf"] if args.algo == "all" else [args.algo]

    rf_model = None  # lazy-train RF only if needed

    for algo in algos_to_run:
        if algo == "wknn":
            res = wknn(scan, db, k=args.k)
        elif algo == "knn":
            res = knn(scan, db, k=args.k)
        elif algo == "bayesian":
            res = bayesian(scan, db)
        elif algo == "rf":
            if rf_model is None:
                rf_model = RandomForestPositioner()
                rf_model.train(raw, all_bssids)
            res = rf_model.predict(scan)

        print_result(res)

if __name__ == "__main__":
    main()