# 📍 Indoor WiFi Localization System

A real-time indoor positioning and navigation system using WiFi RSS fingerprinting. Built as a BTP (B.Tech Project) for comparing machine learning approaches to indoor localization.

## 🏗️ Architecture

The system consists of:
- **Android App** — Scans nearby WiFi access points and sends RSSI data to the server
- **Flask Server** — Processes scans using ML models and serves a real-time navigation dashboard
- **Navigation Dashboard** — Interactive map with live positioning, pathfinding, and turn-by-turn directions

## 🧠 Algorithms Implemented

| Algorithm | Server | Port | Description |
|-----------|--------|------|-------------|
| **Cosine KNN (K=3)** | `app_knn.py` | 5001 | TF-IDF weighted cosine similarity with probabilistic map — **Best accuracy** |
| **Random Forest** | `app_rf.py` | 5002 | Dual Classifier + Regressor with EMA smoothing |
| **WKNN** | `indoor_nav.py` | CLI | Weighted K-Nearest Neighbors (inverse-distance weights) |
| **Bayesian** | `indoor_nav.py` | CLI | Gaussian probabilistic model |

## 📁 Project Structure

```
├── app_knn.py              # KNN server with live dashboard (Port 5001)
├── app_rf.py               # Random Forest server with live dashboard (Port 5002)
├── indoor_nav.py           # Core algorithms (WKNN, KNN, Bayesian, RF)
├── build_prob_map.py       # Generates probabilistic_map.pkl from fingerprints
├── benchmark.py            # Cross-validation benchmarking suite
├── knn.py                  # Cosine KNN engine
├── fingerprints_imu.json   # Primary fingerprint dataset (912 samples, 136 locations)
├── probabilistic_map.pkl   # Pre-computed probabilistic map for KNN
└── floorplan.png           # Building floor plan reference
```

## 🚀 Quick Start

### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask numpy scikit-learn
```

### Run KNN Server (Recommended)
```bash
python3 app_knn.py
```
Open `http://localhost:5001` on your phone (same WiFi network).

### Run Random Forest Server
```bash
python3 app_rf.py
```
Open `http://localhost:5002` on your phone.

### Run Benchmarks
```bash
python3 indoor_nav.py --benchmark --k 3
```

## 📊 Results

| Metric | Cosine KNN (K=3) | Random Forest | WKNN (K=5) | Bayesian |
|--------|:-:|:-:|:-:|:-:|
| Mean Error | **1.42m** | 4.27m | 2.18m | 3.51m |
| Median Error | **0.89m** | 3.12m | 1.45m | 2.87m |
| < 2m Accuracy | **78.4%** | 34.1% | 62.3% | 41.2% |

## ✨ Features

- **Real-time positioning** with sub-second updates
- **Dijkstra pathfinding** with turn-by-turn directions
- **Arrival validation** — confirms location reached after 10 consecutive readings within 2.5m
- **EMA position smoothing** for stable dot movement
- **Dark-mode glassmorphism UI** optimized for mobile
- **Pinch-to-zoom & pan** on the interactive map

## 📝 License

This project was developed as part of a B.Tech thesis at the university.
