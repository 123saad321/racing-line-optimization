# Transformer-Based Racing Line Prediction
### A Deep Learning Approach to Minimum Curvature Path Approximation

**AbuZar Saeed · Ehtisham Ali · Muhammad Ahzam · Muhammad Saad**  
Pakistan Navy Engineering College (PNEC), NUST Karachi  
CS-326: Applied AI and Machine Learning — Spring 2026

---

## Overview

This project develops a two-phase pipeline to predict optimal racing lines for energy-efficient vehicle competition (Shell Eco-Marathon).

**Phase 1** uses the Minimum Curvature Path (MCP) method — a classical L-BFGS-B optimizer — to generate ground truth racing lines from GPS centerline data. The optimal racing line minimizes total path curvature, which reduces braking and acceleration events and therefore reduces fuel consumption.

**Phase 2** trains a Transformer neural network on the Phase 1 outputs to predict racing lines instantly from track geometry alone — without running the optimizer.

The trained model achieves a mean prediction error of **60.9 cm** across 10 unseen test circuits, running in **236 milliseconds** — 28× faster than the MCP optimizer.

---

## Key Results

| Metric | Value |
|---|---|
| Mean MAE (10 test circuits) | **60.9 cm** |
| Improvement over centerline baseline | **63.7%** |
| Inference speed | **236 ms** |
| Speed vs MCP optimizer | **28× faster** |
| Best circuit — Circuit de Nevers Magny-Cours | 33 cm |
| Worst circuit — Las Vegas Street Circuit | 132 cm |
| Model size | ~0.4 MB |

### Per-Circuit Results

| Circuit | MAE (cm) |
|---|---|
| Circuit de Nevers Magny-Cours | 33 |
| Circuit de Monaco | 37 |
| Autódromo Oscar y Juan Gálvez | 37 |
| Nürburgring | 43 |
| Autódromo Internacional do Algarve | 49 |
| Autodromo Nazionale Monza | 50 |
| Indianapolis Motor Speedway | 62 |
| Yas Marina Circuit | 67 |
| Jeddah Corniche Circuit | 98 |
| Las Vegas Street Circuit | 132 |

---

## Repository Structure

```
racing-line-optimization/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── geojson/                    ← 40 raw GeoJSON circuit files
│   └── circuits_summary.csv        ← Metadata: name, location, length for all 40 circuits
│
├── ground_truth/
│   ├── train/                      ← 30 MCP ground truth CSVs (Phase 1 output, Phase 2 input)
│   └── test/                       ← 10 held-out test circuit CSVs (never seen during training)
│
├── notebooks/
│   ├── Phase1_Ground_Truth_Generation.ipynb   ← GeoJSON → MCP racing line → ground truth CSV
│   ├── Phase2_Building_Model.ipynb            ← Feature engineering → Transformer training
│   ├── Validation_Alpha_Comparison.ipynb      ← Predicted vs ground truth plots for 10 test tracks
│   └── Comparative_Analysis_Final.ipynb       ← Side-by-side MCP vs Transformer comparison
│
├── results/
│   ├── Predictions.json            ← Best run predictions for all 10 test circuits
│   ├── Racing_Line_Transformer.pth ← Trained model weights (best run)
│   └── training_curve.png          ← Training and validation loss curve
│
├── figures/
│   ├── generate_figures.py         ← Generates all 10 presentation figures
│   ├── CIRCUI_2.csv                ← Magny-Cours MCP ground truth (used by figure generator)
│   ├── F1_world_map.png            ← Dataset world map (40 circuits, 29 countries)
│   ├── F2_pipeline.png             ← End-to-end pipeline diagram
│   ├── F3_mcp_racing_line.png      ← MCP racing line on Magny-Cours
│   ├── F4_curvature.png            ← Curvature map and feature visualization
│   ├── F5_architecture_window.png  ← Transformer architecture and sliding window
│   ├── F6_training_curve.png       ← Training dynamics
│   ├── F7_all_tracks.png           ← MAE bar chart for all 10 test circuits
│   ├── F8_best_segment.png         ← Zoomed best segment (pts 58–118, MAE = 3 cm)
│   ├── F9_worst_segment.png        ← Zoomed worst segment (pts 297–357, MAE = 117 cm)
│   └── F10_tradeoff.png            ← MCP vs Transformer trade-off summary
│
└── literature/
    ├── Batavia_Gasoline_Team_Report.pdf     ← Prior SEM team on-track validation (Shell website)
    ├── Bayesian_RacingLine.pdf
    ├── Genetic_RacingLine.pdf
    ├── Multilayer_Graph_Trajectory.pdf
    ├── NMPC_AutonomousRacing.pdf
    ├── Racing_Line_Optimization_MIT.pdf
    ├── RealTime_ML_Trajectory.pdf
    └── Sequential_TwoStep_Algorithm.pdf
```

---

## Pipeline

```
GeoJSON (bacinger/f1-circuits)
    ↓  convert_circuits.py — parse coordinates
    ↓  Resample to 800 uniform points
    ↓  Savitzky-Golay smoothing (window=15, poly=3)
    ↓  Wall generation ±4.5 m perpendicular to centerline
    ↓  MCP optimizer (L-BFGS-B) — minimise Σκ² + λΣ(Δα)²
Ground truth CSV: [t_lon, t_lat, alpha]          ← Phase 1 output
    ↓  Feature engineering → 7 features per point
    ↓  Left-right mirroring augmentation (30 → 60 circuits)
    ↓  RacingLineTransformer — sliding window seq_len=21
    ↓  Loss = MAE + λ·max(0, min_std − std(α̂))
Predicted alpha per circuit — 236 ms inference   ← Phase 2 output
```

---

## Model Architecture

```
Input:  (batch, seq_len=21, features=7)
        Features: local_x, local_y, delta_heading,
                  curvature (κ), dist_inner, dist_outer, cum_dist_norm

Linear Projection  →  d_model = 64

× 3  TransformerEncoderLayer
     (n_heads=2, dim_feedforward=128, dropout=0.1)

Linear(64 → 32) → ReLU → Linear(32 → 1) → Sigmoid

Output: α̂ ∈ (0, 1)   [0 = inner wall, 1 = outer wall]
```

**Loss function:**
```
Loss = MAE(α̂, α) + λ · max(0, min_std − std(α̂))
       λ = 2.0,  min_std = 0.05
```
The variance penalty prevents the model from collapsing to α ≈ 0.5 (centerline) on every prediction.

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR schedule | CosineAnnealingLR |
| Epochs | 120 |
| Batch size | 256 |
| Training circuits | 30 original + 30 mirrored = 60 |
| Test circuits | 10 (fixed holdout, never seen during training) |

---

## How to Run

All notebooks are designed to run in **Google Colab**. Open each notebook, follow the steps, and upload the required files when prompted.

### Step 1 — Generate Ground Truth (Phase 1)

Open `notebooks/Phase1_Ground_Truth_Generation.ipynb` in Colab.

Upload any circuit CSV from `data/geojson/` (after converting with `convert_circuits.py`), or use an existing CSV from `ground_truth/`. The notebook outputs a ground truth CSV with columns `[t_lon, t_lat, alpha]`.

### Step 2 — Train the Transformer (Phase 2)

Open `notebooks/Phase2_Building_Model.ipynb` in Colab.

Upload all 40 ground truth CSVs from `ground_truth/train/` and `ground_truth/test/`. The notebook trains the model and saves `predictions.json` and `Racing_Line_Transformer.pth`.

### Step 3 — Validate Results

Open `notebooks/Validation_Alpha_Comparison.ipynb` in Colab.

Upload `results/Predictions.json` and the 10 test CSVs from `ground_truth/test/`. The notebook generates per-track accuracy metrics and racing line plots.

### Step 4 — Run Comparative Analysis

Open `notebooks/Comparative_Analysis_Final.ipynb` in Colab.

Upload `results/Racing_Line_Transformer.pth`, `results/Predictions.json`, and any one test circuit CSV. The notebook runs MCP and Transformer side by side and measures computation time.

### Step 5 — Generate Presentation Figures

```bash
cd figures/
pip install matplotlib numpy scipy pandas geopandas cartopy pyogrio
python generate_figures.py
```

Place `CIRCUI_2.csv` (already in the folder) and `results/Predictions.json` in the same directory. All 10 figures are saved to `figures/figures/`.

---

## Dataset

**Source:** [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) — open-source GPS coordinates of racing circuits.

| Property | Value |
|---|---|
| Total circuits | 40 |
| Countries | 29 |
| Training set | 30 circuits (×2 with mirroring = 60) |
| Test set | 10 circuits (fixed holdout) |
| Points per circuit | 800 (uniformly resampled) |
| Assumed track width | 9 metres |

**Test circuits:** Autódromo Internacional do Algarve, Autódromo Oscar y Juan Gálvez, Autodromo Nazionale Monza, Circuit de Monaco, Circuit de Nevers Magny-Cours, Indianapolis Motor Speedway, Jeddah Corniche Circuit, Las Vegas Street Circuit, Nürburgring, Yas Marina Circuit.

**Limitation:** All circuits are professional racing tracks (>4 km, high-speed). Shell Eco-Marathon circuits are typically shorter and more compact. Generalization to SEM-specific layouts has not been validated.

---

## Dependencies

```
torch
numpy
pandas
scipy
scikit-learn
matplotlib
geopandas
cartopy
pyogrio
simplekml
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

1. Heilmeier, A., et al. (2020). Minimum curvature trajectory planning and control for an autonomous race car. *Vehicle System Dynamics*, 58(10), 1497–1527.
2. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.
3. Betz, J., et al. (2022). A software framework for autonomous motorsport applications. *IEEE Transactions on Intelligent Vehicles*.
4. Christ, F., et al. (2021). Time-optimal trajectory planning for a race car considering variable tyre-road friction coefficients. *Vehicle System Dynamics*, 59(4), 588–612.
5. Bacinger, R. (2020). f1-circuits. GitHub. https://github.com/bacinger/f1-circuits
6. Batavia Gasoline Team (2025). OTA Data and Telemetry — Shell Eco-Marathon Asia Pacific & Middle East. Shell Eco-Marathon Official Publication.

---

## Acknowledgements

Supervised by [Supervisor Name], Pakistan Navy Engineering College (PNEC), NUST Karachi.  
Raw circuit data sourced from [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits).

---

*CS-326: Applied AI and Machine Learning — Spring 2026*  
*Pakistan Navy Engineering College (PNEC), NUST Karachi*
