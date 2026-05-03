# Wireless Indoor Localization and Navigation System

**Using WiFi RSSI Fingerprinting with Multiple Localization Models**

*Course Project Report — Wireless Networks (WN)*
*4th Floor, R&D Building Deployment*

---

## Abstract

Indoor localization remains a challenging problem as GNSS (GPS) is ineffective indoors due to signal attenuation and multipath propagation through walls and ceilings. This project presents a complete end-to-end wireless indoor localization and navigation system deployed on the **4th floor of the R&D building**.

The system employs **WiFi Received Signal Strength Indicator (RSSI) fingerprinting** as its primary sensing modality. We implement and deploy two distinct localization models, each with its own Flask-based navigation server:

1. **Model A — TF-IDF Cosine Similarity KNN**: Uses IDF-weighted cosine similarity over a probabilistic radio map, achieving a best mean LOLO error of **3.51 m** across 27 candidate configurations (*K = 3, γ = 4*).

2. **Model B — Stacked Random Forest + Gradient Boosting Ensemble**: Employs a two-layer stacking architecture with 300-tree Random Forest and 200-tree Gradient Boosting base learners, a Ridge meta-learner, and engineered RSSI features (MinMax normalization, presence masks, visibility statistics).

Both models share a common navigation subsystem that represents the floor as a weighted graph, applies **Dijkstra's shortest-path algorithm**, and generates **turn-by-turn navigation instructions**. Position estimates are stabilized through a confidence-adaptive **Exponential Moving Average (EMA)** filter with history averaging.

**Keywords**: WiFi fingerprinting, indoor localization, TF-IDF, cosine similarity, KNN, Random Forest, Gradient Boosting, stacking ensemble, Dijkstra, Flask, RSSI

---

## System Architecture

The system is divided into two temporal phases and three functional subsystems:

```
Data Collection → parse_fingerprints.py → build_prob_map.py → probabilistic_map.pkl
                                                                        ↓
Client WiFi Scan → /api/locate (Flask) → CosineKNN / RF predict() → EMA Filter → Nearest Node Snap
                                                                                         ↓
                                                              /api/navigate → Dijkstra Routing → Web Dashboard
```

### Offline Training Phase
Team members physically visited **136 reference locations** across the 4th floor. At each location, a WiFi scanning tool recorded the BSSID and RSSI of each visible access point, together with *(x, y)* coordinates and a human-readable label. Multiple scans were collected across separate sessions to capture temporal variability (**912 total samples**, **500 unique BSSIDs**).

### Online Inference Phase
During live operation, the Android client sends a raw `{BSSID: RSSI}` dictionary to `/api/locate`. The backend invokes the active model, feeds the result through the multi-stage smoothing pipeline, and returns the estimated *(x, y)*, nearest graph node, confidence score, and filter metadata.

### Web Dashboard Frontend
The Flask server serves an interactive **HTML5 Canvas dashboard** with:
- Pan (drag) and zoom (scroll / pinch) support
- Animated blue dot using LERP interpolation at 12% per `requestAnimationFrame` tick
- Active route highlighted in red
- Turn-by-turn directions below the map
- Auto-polling at **500 ms** intervals with no page reload

---

## Methodology

### Coordinate System & Floor Graph
A metric coordinate system is defined with the **elevator at the origin (0, 0)**. All distances are in meters, measured physically by walking the floor.

The navigation graph *G = (V, E)* is automatically constructed from the fingerprint dataset. Two layers of physical constraints prevent wall-crossing routes:
1. **Blocked node list**: structural obstacles (pillars) excluded entirely
2. **Blocked edge list**: 30+ explicitly forbidden node pairs preventing edges through walls, rooms, and impassable segments

Each node connects to at most **3 nearby neighbors** within *d_max = 4.0 m*.

### WiFi RSSI Fingerprint Data Collection
- **BSSID** is used exclusively (not SSID) because all institutional APs share common SSIDs like `eduroam`
- Each physical AP broadcasts **5–6 virtual BSSIDs** across radio bands
- Missing BSSIDs in a scan are filled with a sentinel of **−100 dBm**

### Localization Algorithms

#### Model A: TF-IDF Cosine Similarity KNN
Addresses two key problems:
- **Cross-device magnitude variation**: Cosine similarity is invariant to overall signal strength scaling
- **Uninformative ubiquitous APs**: IDF weighting automatically suppresses floor-wide APs

```
IDF(a) = log(N / (df(a) + 1))
v_ℓ,a = (μ_ℓ,a − r_min) · IDF(a)
cos(q', v_ℓ) = q̂' · v̂_ℓ
```

The top-*K* locations are selected and the predicted position is their exponentially-sharpened similarity-weighted centroid (*K = 3, γ = 4*).

#### Model B: Stacked Random Forest + Gradient Boosting Ensemble
- **Base models**: Two 300-tree RF regressors + two 200-tree GB regressors
- **Feature engineering**: MinMax-normalized RSSI, binary presence mask, element-wise product, visible AP count, mean/std RSSI
- **Meta-learner**: Ridge regression trained on Out-Of-Fold predictions (5-fold CV)
- **Optional IMU dead reckoning** for position stabilization between scans

### Position Smoothing Pipeline

```
Raw Prediction → First-Fix Init → Confidence-Adaptive EMA → History Averaging → Nearest Node Snap
                                     α_eff = α₀ · c           last 5 predictions
                                     jump damp if > 8m         0.6 EMA + 0.4 hist
```

- **α₀ = 0.35** balances responsiveness and stability
- **δ_max = 8.0 m** jump threshold prevents teleportation from outlier predictions
- **Arrival validation**: Location confirmed only after **10 consecutive readings** within 2.5 m

---

## Project Structure

```
├── app_knn.py              # Model A server with live dashboard (Port 5001)
├── app_rf.py               # Model B server with live dashboard (Port 5002)
├── indoor_nav.py           # Core algorithms (WKNN, KNN, Bayesian, Random Forest)
├── build_prob_map.py       # Generates probabilistic_map.pkl from fingerprints
├── benchmark.py            # Cross-validation benchmarking suite
├── knn.py                  # Cosine KNN engine with TF-IDF weighting
├── fingerprints_imu.json   # Primary fingerprint dataset (912 samples, 136 locations)
├── probabilistic_map.pkl   # Pre-computed probabilistic radio map
├── floorplan.png           # Building floor plan reference
└── WN_Project.pdf          # Full project report
```

---

## Quick Start

### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install flask numpy scikit-learn
```

### Run Model A — Cosine KNN Server (Recommended)
```bash
python3 app_knn.py
```
Dashboard: `http://localhost:5001`

### Run Model B — Random Forest Server
```bash
python3 app_rf.py
```
Dashboard: `http://localhost:5002`

### Run Benchmarks
```bash
python3 indoor_nav.py --benchmark --k 3
```

> **Note**: Open the dashboard URL on your phone (same WiFi network) for real-time localization.

---

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/locate` | POST | Submit WiFi scan for localization |
| `/api/current` | GET | Get current estimated position |
| `/api/navigate` | POST | Get Dijkstra path between two nodes |
| `/api/graph` | GET | Full navigation graph (nodes + edges) |
| `/api/nodes` | GET | Navigable destination list |
| `/api/reset` | POST | Reset position state |

---

## Challenges & Lessons Learned

- **Android WiFi throttling**: API 28+ limits scans to 4 per 2-minute window; EMA smoothing mitigates this
- **BSSID vs SSID**: All institutional APs share SSIDs (`eduroam`), making SSID completely uninformative; BSSID is essential
- **Wall-crossing routes**: Naively connecting nearby nodes produced impossible paths; resolved with dual-constraint system (30+ blocked edges)
- **TF-IDF insight**: Ubiquitous APs contributed noise until IDF weighting suppressed them, shifting attention to location-specific APs
- **Signal aliasing at A-420**: Localized degradation due to structural interference and multipath fading; highlights need for additional modalities like BLE beacons

---

## Future Work

- Incremental map updates to reduce recalibration overhead
- Additional modalities (BLE beacons, visual odometry) fused with WiFi RSSI
- Automatic blocked-edge detection from architectural floor plans
- Multi-floor navigation with elevator state detection

---

## Team

| Member | Contribution |
|--------|-------------|
| **Piyush Aggarwal** | Dataset collection; Cosine KNN similarity model |
| **Arvind Dhavala** | Dataset collection; KNN + IMU model |
| **Anant Gyan Singhal** | Dataset collection; Random Forest model |
| **Saksham Kakkar** | Dataset collection; Unweighted Euclidean KNN (K=3, K=5) |
| **Harshit Tandon** | Dataset collection; WKNN model |

*Note: The application backend and web dashboard frontend were partially developed using AI assistance.*
