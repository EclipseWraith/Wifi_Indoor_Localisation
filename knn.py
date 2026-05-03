#!/usr/bin/env python3
"""
Cosine Similarity KNN Localization Model
-----------------------------------------
Uses cosine similarity between WiFi RSSI vectors for indoor localization.
Winner of the 27-model LOLO comparison with 3.51m mean error.

Cosine similarity is robust to overall signal strength variations between
phones — it measures the angle (direction) of the RSSI vector, not magnitude.
"""

import json
import math
import pickle
import numpy as np
from collections import defaultdict

DEFAULT_RSSI = -100
K_NEIGHBORS = 3
EXP_WEIGHT = 4.0

class CosineKNN:
    def __init__(self, map_path="probabilistic_map.pkl"):
        with open(map_path, "rb") as f:
            self.p_map = pickle.load(f)

        self.locations = list(self.p_map.keys())

        # Build BSSID index and compute Document Frequencies (for TF-IDF)
        doc_freq = defaultdict(int)
        for loc_data in self.p_map.values():
            for bssid in loc_data["aps"].keys():
                doc_freq[bssid] += 1
                
        self.bssid_list = sorted(doc_freq.keys())
        self.bssid_idx = {b: i for i, b in enumerate(self.bssid_list)}
        self.n_feats = len(self.bssid_list)
        
        # Calculate Inverse Document Frequency (IDF) array
        total_docs = len(self.locations)
        self.idf = np.zeros(self.n_feats, dtype=np.float64)
        for bssid, freq in doc_freq.items():
            self.idf[self.bssid_idx[bssid]] = math.log(total_docs / (freq + 1))

        # Precompute reference vectors (mean RSSI per location, shifted to positive, scaled by IDF)
        self.ref_vectors = np.zeros((len(self.locations), self.n_feats), dtype=np.float64)
        self.ref_coords = np.zeros((len(self.locations), 2), dtype=np.float64)
        self.ref_labels = []

        for i, loc in enumerate(self.locations):
            vec = np.zeros(self.n_feats, dtype=np.float64)
            for bssid, stats in self.p_map[loc]["aps"].items():
                if bssid in self.bssid_idx:
                    idx = self.bssid_idx[bssid]
                    vec[idx] = (stats["mu"] - DEFAULT_RSSI) * self.idf[idx]
            self.ref_vectors[i] = vec
            self.ref_coords[i] = [loc[0], loc[1]]
            self.ref_labels.append(self.p_map[loc]["label"])

        # Precompute norms
        self.ref_norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
        self.ref_norms[self.ref_norms == 0] = 1.0
        self.ref_unit = self.ref_vectors / self.ref_norms

        print(f"[CosineKNN] {len(self.locations)} locations, {self.n_feats} BSSIDs, K={K_NEIGHBORS}")

    def predict(self, scan, heading_deg=0.0):
        """Predict (x, y) from a live WiFi scan dict {bssid: rssi}."""
        if not scan:
            return {"x": 0, "y": 0, "label": "no scan", "confidence": 0}

        # Build scan vector
        test_vec = np.zeros(self.n_feats, dtype=np.float64)
        for bssid, rssi in scan.items():
            if bssid in self.bssid_idx:
                idx = self.bssid_idx[bssid]
                test_vec[idx] = (float(rssi) - DEFAULT_RSSI) * self.idf[idx]
                
        tn = np.linalg.norm(test_vec)
        if tn == 0:
            return {"x": 0, "y": 0, "label": "no signal", "confidence": 0}
        test_unit = test_vec / tn

        # Cosine similarities with ALL reference locations (vectorized)
        cos_sims = self.ref_unit @ test_unit  # (N_locs,)

        # Get top-K
        topk_idx = np.argpartition(-cos_sims, K_NEIGHBORS)[:K_NEIGHBORS]
        topk_sims = cos_sims[topk_idx]
        topk_coords = self.ref_coords[topk_idx]

        # Weighting by similarity with exponential smoothing
        weights = np.maximum(topk_sims, 1e-6) ** EXP_WEIGHT
        total_w = weights.sum()
        pred = (weights[:, None] * topk_coords).sum(axis=0) / total_w

        # Label from best match
        best_idx = topk_idx[np.argmax(topk_sims)]
        label = self.ref_labels[best_idx]

        # Confidence from similarity spread
        confidence = float(np.max(topk_sims))

        return {
            "x": round(float(pred[0]), 2),
            "y": round(float(pred[1]), 2),
            "label": label,
            "confidence": round(confidence, 3),
        }
