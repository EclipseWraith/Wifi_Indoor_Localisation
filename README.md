# 📍 WiFi Indoor Localization & Navigation

A real-time indoor positioning and navigation system using WiFi RSSI fingerprinting. The system compares multiple machine learning approaches — **TF-IDF Cosine KNN**, **Random Forest**, **WKNN**, and **Bayesian** — and deploys the best models as Flask web servers with interactive map dashboards.

> GPS doesn't work indoors. This system uses the WiFi signals already around you to determine your location with meter-level accuracy.

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│ Android App │────▶│ Flask Server │────▶│  ML Prediction │────▶│  Web Dashboard  │
│  WiFi Scan  │     │  /api/locate │     │  + EMA Filter  │     │  Live Map + Nav │
└─────────────┘     └──────────────┘     └────────────────┘     └─────────────────┘
```

1. **Offline Phase**: Walk through the building collecting WiFi scans at known locations → builds a fingerprint database
2. **Online Phase**: Phone sends a live WiFi scan → server predicts your (x, y) position using the trained model → dashboard shows your location on an interactive map
3. **Navigation**: Select a destination → Dijkstra's algorithm computes the shortest walkable path → turn-by-turn directions guide you there

---

## Benchmark Results

5-fold cross-validation on 912 samples across 136 locations (500 unique BSSIDs):

| Algorithm | Mean Error | Median Error | < 2 unit Accuracy |
|-----------|:---------:|:------------:|:-----------------:|
| **Random Forest** | **3.36** | **2.21** | **44.5%** |
| Bayesian | 6.47 | 4.24 | 21.5% |
| WKNN (K=3) | 6.28 | 4.27 | 21.2% |
| KNN (K=3) | 6.86 | 4.74 | 17.1% |

*Run `python3 indoor_nav.py --benchmark --k 3` to reproduce.*

The **Cosine KNN** model (served by `app_knn.py`) uses a separate TF-IDF weighted probabilistic map with exponential similarity sharpening, achieving best real-world performance due to its robustness against cross-device signal strength variation.

---

## Features

- **Two deployable servers** — Cosine KNN (`app_knn.py`, port 5001) and Random Forest (`app_rf.py`, port 5002)
- **Real-time positioning** with confidence-adaptive EMA smoothing
- **Interactive map** — pan, zoom, pinch-to-zoom on mobile
- **Dijkstra navigation** with turn-by-turn directions
- **Arrival validation** — confirms destination reached after 10 consecutive readings within 2.5m
- **Graph constraints** — 30+ blocked edges prevent routes through walls
- **Dark-mode UI** — glassmorphism dashboard optimized for mobile browsers

---

## Project Structure

```
├── app_knn.py              # Cosine KNN server + dashboard (port 5001)
├── app_rf.py               # Random Forest server + dashboard (port 5002)
├── indoor_nav.py           # All algorithms: WKNN, KNN, Bayesian, Random Forest
├── knn.py                  # TF-IDF Cosine KNN engine
├── build_prob_map.py       # Builds probabilistic radio map from fingerprints
├── benchmark.py            # Extended benchmarking suite
├── fingerprints_imu.json   # Fingerprint dataset (912 samples, 136 locations)
├── probabilistic_map.pkl   # Pre-computed probabilistic map for Cosine KNN
└── floorplan.png           # Floor plan reference image
```

---

## Getting Started

### Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask numpy scikit-learn
```

### Run the Cosine KNN server
```bash
python3 app_knn.py
```

### Run the Random Forest server
```bash
python3 app_rf.py
```

Open the dashboard URL on any device connected to the same WiFi network.

### Run benchmarks
```bash
python3 indoor_nav.py --benchmark --k 3
```

### Test a single prediction
```bash
python3 indoor_nav.py --algo rf --demo
python3 indoor_nav.py --algo wknn --demo --k 5
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/locate` | `POST` | Submit `{"networks": {"BSSID": RSSI}}` for localization |
| `/api/current` | `GET` | Current estimated position + confidence |
| `/api/navigate` | `POST` | `{"from": "node", "to": "node"}` → Dijkstra path |
| `/api/graph` | `GET` | Full navigation graph (nodes + edges) |
| `/api/nodes` | `GET` | List of navigable destinations |
| `/api/reset` | `POST` | Reset position state |

---

## Key Design Decisions

- **BSSID over SSID** — Institutional APs share common SSIDs (e.g., `eduroam`), making SSID useless for localization. BSSID uniquely identifies each radio interface.
- **TF-IDF weighting** — Suppresses ubiquitous APs visible floor-wide that add noise, boosting location-specific APs that carry discriminative signal.
- **Cosine similarity** — Invariant to absolute signal magnitude, making the model robust across different phone hardware.
- **Position smoothing** — Three-stage pipeline (first-fix → confidence-adaptive EMA → history averaging) prevents jitter and outlier teleportation.
- **Arrival debounce** — Requires 10 consecutive sub-2.5m readings before confirming arrival, preventing false positives from signal spikes.

---

## Adapting to Your Building

1. **Collect fingerprints**: Walk through your building, recording WiFi scans at known (x, y) locations. Save as JSON with format:
   ```json
   [{"x": 5.0, "y": 3.2, "label": "Room 101", "networks": {"AA:BB:CC:DD:EE:FF": -65, ...}}, ...]
   ```
2. **Build the probabilistic map**: `python3 build_prob_map.py`
3. **Update graph constraints**: Edit `BLOCKED_EDGE_PATTERNS` in the server file to match your building's walls
4. **Run the server**: `python3 app_knn.py`

---

## License

MIT
