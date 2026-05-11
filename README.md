# 📡 Project 7 — Multipath Analysis

> **MP1 · MP2 · Arc Splitting · RMS Ranking · Sidereal Repeat | GPS | Auckland, NZ**

---

## 📌 Overview

Multipath is one of the most challenging GNSS error sources — it cannot be modelled
from first principles because it depends entirely on the local environment around the
receiver antenna. This project computes the **MP1 and MP2 multipath proxies** for
all GPS satellites and performs a full site characterisation.

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| RMS(MP1) < 0.30 m | ✅ Clean | Low multipath environment |
| RMS(MP1) 0.30–0.50 m | ⚠️ Moderate | Some nearby reflectors |
| RMS(MP1) > 0.50 m | ❌ High | Significant multipath |

---

## 📐 Key Equations

### MP1 — L1 code multipath proxy:
```
MP1 = C1C − Φ₁ − α·(Φ₁ − Φ₂)
    where α = 2f₂²/(f₁²−f₂²) ≈ 1.5457
```

### MP2 — L2 code multipath proxy:
```
MP2 = C2W − Φ₂ − β·(Φ₁ − Φ₂)
    where β = 2f₁²/(f₁²−f₂²) ≈ 2.5457
```

### After arc-mean removal:
```
MP1 − mean(MP1_arc) ≈ pure multipath + receiver noise
```

### Sidereal repeat:
```
Δt_sidereal = 24h − 23h 56m 4.1s ≈ 3m 56s per day
```
Multipath repeats every sidereal day — shifted 3m 56s earlier in solar time.

---

## 🖼️ Output Plots

All plots are saved in `output/`.

### Plot 1 — MP1 Time Series
Top-8 satellites by arc length. Oscillations = multipath varying as satellite moves.
Reference lines at ±0.30 m and ±0.50 m quality thresholds.

### Plot 2 — MP1 Heatmap
Diverging red/blue colourmap:
- **Red** = positive multipath (code longer than phase-derived range)
- **Blue** = negative multipath
- **Dark** = satellite not tracked

Vertical bands that appear across multiple satellites at the same time of day
indicate a site-specific reflector active in that direction.

### Plot 3 — MP1 RMS Ranking
Bar chart of all GPS satellites sorted by RMS(MP1), colour-coded:
- Green = clean, Orange = moderate, Red = high multipath

### Plot 4 — MP Distribution Histogram
MP1 and MP2 distributions with Gaussian fit.
A narrow, symmetric distribution = clean environment.
Asymmetry or heavy tails = systematic reflectors.

### Plot 5 — Sidereal Repeat
Demonstrates the sidereal repeat concept: multipath shifts ~3m 56s earlier
each solar day, enabling sidereal filtering for multipath mitigation.

---

## 📂 File Structure

```
project_8/
├── src/
│   └── project8_multipath.ipynb   ← Main notebook
├── output/                         ← Generated plots (auto-filled on run)
│   ├── plot1_mp1_timeseries.png
│   ├── plot2_mp1_heatmap.png
│   ├── plot3_mp1_rms_ranking.png
│   ├── plot4_mp_histogram.png
│   └── plot5_sidereal_repeat.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure the shared utils are accessible
The notebook imports from `../../shared/utils/gnss_utils.py`.
Run from inside `src/` or adjust the path.

### 3. Set your RINEX file path
In **Step 2** of the notebook:
```python
obs_path = "../../data/AUCK00NZL_R_20260010000_01D_30S_MO.rnx"
```

### 4. Run all cells
```bash
jupyter notebook src/project8_multipath.ipynb
```

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `georinex` | Parse RINEX 3 observation files |
| `xarray` | N-dimensional labelled arrays |
| `pandas` | Time series manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Publication-quality plotting |

---

## 👤 Author

**Hakim El Azzouzi**
MSc Global Navigation Satellite Systems
Mohammed First University, Oujda, Morocco
📧 elazzouzihakim10@gmail.com
🔗 [linkedin.com/in/Hakim-El-Azzouzi](https://linkedin.com/in/Hakim-El-Azzouzi)
📍 Luxembourg 🇱🇺

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 GNSS Observation RINEX Series

| # | Project |
|---|---------|
| 1 | Single GPS Satellite — Pseudorange & SNR Heatmap |
| 2 | All GPS Satellites — Fleet Pseudorange & SNR Heatmap |
| 3 | Multi-Constellation GNSS — One Satellite per System |
| 4 | Pseudorange vs Carrier-Phase Comparison |
| 5 | Constellation Summary — Pie Chart & Histograms |
| 6 | Ionospheric Delay — Geometry-Free Combination |
| **7** | **Multipath Analysis** ← You are here |
| 8 | Data Quality Report |

