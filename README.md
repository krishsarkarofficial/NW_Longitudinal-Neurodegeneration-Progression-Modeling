# OASIS-2 Longitudinal Neurodegeneration Progression Modeling & 3D Brain Digital Twin

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-OASIS--2%20DUA-green.svg)](https://www.oasis-brains.org/)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](tests/)

An end-to-end neuroinformatics and deep learning platform for forward-simulating Alzheimer's Disease progression. Combining structural neuroimaging parcellation, biophysical network diffusion over axonal connectomes, spatio-temporal deep learning (Graph Neural ODEs, Physics-Constrained VAEs, and Diffusion Models), clinical cognitive deficit mapping, and an interactive **Standalone 3D WebGL Brain Digital Twin**.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [System Architecture & Pipeline Flow](#system-architecture--pipeline-flow)
- [Prerequisites & Installation](#prerequisites--installation)
- [How to Run the Pipeline](#how-to-run-the-pipeline)
  - [1. Quick Start Execution (CLI)](#1-quick-start-execution-cli)
  - [2. Custom Execution Flags](#2-custom-execution-flags)
  - [3. Running the Test Suite](#3-running-the-test-suite)
  - [4. Python API Usage](#4-python-api-usage)
- [How to Run & View the 3D WebGL Digital Twin Simulation](#how-to-run--view-the-3d-webgl-digital-twin-simulation)
  - [1. Launching the Simulation](#1-launching-the-simulation)
  - [2. Rebuilding the Simulation from Model Checkpoints](#2-rebuilding-the-simulation-from-model-checkpoints)
  - [3. Interactive Simulation Controls](#3-interactive-simulation-controls)
- [Repository Structure](#repository-structure)
- [Technical Documentation Index](#technical-documentation-index)
- [Key Methodology & Leakage Prevention](#key-methodology--leakage-prevention)
- [References](#references)

---

## Executive Overview

Neurodegenerative diseases such as Alzheimer's Disease (AD) do not spread randomly across the brain; pathology spreads along anatomical axonal highways defined by the brain's structural connectome. Cross-sectional diagnostics only provide a static snapshot of damage that has already occurred.

This platform addresses two core clinical questions:
1. **What brain regions are currently degenerated?** Quantified using automated 56-region SynthSeg parcellation normalized for estimated Total Intracranial Volume (eTIV) and residualized against healthy aging via Ordinary Least Squares ($W$-scores).
2. **Where will the pathology spread next, and what cognitive faculties will decline?** Forward-projected across 5-year horizons using biophysical heat-kernel graph diffusion and spatio-temporal deep learning models, then mapped directly into 11 clinical cognitive domains (Episodic Memory, Executive Function, Language, etc.).

---

## System Architecture & Pipeline Flow

The computational pipeline consists of eight integrated phases:

```text
+------------------------------------------------------------------------------------------------+
|                            END-TO-END SYSTEM PIPELINE ARCHITECTURE                             |
+------------------------------------------------------------------------------------------------+

  [STREAM 1: NEUROIMAGING & VOLUMETRICS]                [STREAM 2: STRUCTURAL CONNECTOME]
  
  +----------------------------------------+            +----------------------------------------+
  |       Raw T1 MRI Scans (OASIS-2)       |            |     Structural Connectome Matrix A     |
  |  Longitudinal Cohort: Dementia/Control |            |   Template Matrix (56 x 56 Parcels)    |
  +-------------------+--------------------+            +-------------------+--------------------+
                      |                                                     |
                      v                                                     v
  +----------------------------------------+            +----------------------------------------+
  |     Phase 0 & 1: Data Ingestion &      |            |        Phase 2: Graph Laplacian        |
  |         SynthSeg Parcellation          |            |        L = D - A,  D_ii = sum(A_ij)    |
  |      (56 Desikan-Killiany Parcels)     |            |       L = V diag(lambda) V^T           |
  +-------------------+--------------------+            +-----------+--------------------+-------+
                      |                                             |                    |
                      v                                             |                    |
  +----------------------------------------+                        |                    |
  |       56 Regional Brain Volumes        |                        |                    |
  |     Normalized by eTIV (Vol / eTIV)    |                        |                    |
  +-------------------+--------------------+                        |                    |
                      |                                             |                    |
                      v                                             |                    |
  +----------------------------------------+                        |                    |
  |       Age-Residualized W-Scores        |                        |                    |
  |   W = (Vol_obs - Vol_pred) / sigma_ctrl|                        |                    |
  |     (OLS on Healthy Normal Aging)      |                        |                    |
  +-----------+--------------------+-------+                        |                    |
              |                    |                                |                    |
              |                    +----------------+  +------------+                    |
              |                                     |  |                                 |
              v                                     v  v                                 v
  +----------------------------------------+    +----------------------------------------+
  |  Phase 3: Network Diffusion Model NDM  |    |  Phase 5: Deep Learning Benchmarking   |
  |       x(t) = exp(-beta * L * t) x_0    |    |     Spatio-Temporal Graph Dynamics     |
  |  Analytical Closed-Form Diffusion Soln |    |    Connectome-Informed Neural Models   |
  +-------------------+--------------------+    +-------+----------------+---------------+
                      |                                 |                |               |
                      v                                 v                v               v
  +----------------------------------------+     +-------------+  +-------------+  +-------------+
  |     Phase 4: Cognitive Deficit Map     |     | ST-GNN-ODE  |  |   ND-VAE    |  |    TADM     |
  |  Deficit_d = sum(W_i * w_i,d)/sum(w_i) |     | Continuous  |  |  Physics-   |  | Trajectory- |
  |   11 Functional Domains (Memory, etc.) |     |  Graph ODE  |  | Constrained |  |    Aware    |
  +-------------------+--------------------+     +------+------+  +------+------+  |  Diffusion  |
                      |                                 |                |         +-------------+
                      |                                 v                v
                      |                          +-------------+  +------------------------------+
                      |                          |   Phase 7   |  |    Phase 6: Digital Twin     |
                      |                          |Explainabili-|  |    Autoregressive Rollout    |
                      |                          |     ty      |  | 5-Year Forward Trajectories  |
                      |                          | (Integrated |  |  95% Confidence MC Bands    |
                      |                          |  Gradients) |  +--------------+---------------+
                      |                          +------+------+                 |
                      |                                 |                        |
                      +-------------------+             |                        |
                                          |             |                        |
                                          v             v                        v
+------------------------------------------------------------------------------------------------+
|                   Phase 8: Standalone Interactive 3D WebGL Digital Twin                        |
|                         Target: outputs/digital_twin_3d_simulation.html                        |
|                                                                                                |
|   * Real-Time 3D Anatomical Brain: 56-ROI continuous colormap (Normal -> Severe Atrophy)       |
|   * Interactive Multi-Year Timeline: Slider scrub & auto-play animation (Year 0 -> Year 5+)    |
|   * Dynamic 11-Domain Cognitive Radar Chart: Synchronized clinical deficit progression         |
|   * Multimodal Cohort Comparison: Demented vs. Converted vs. Healthy Non-Demented trajectories |
+------------------------------------------------------------------------------------------------+
```

For full mathematical derivations, consult the [Functional Design Document](docs/FUNCTIONAL_DESIGN_DOCUMENT.md).

---

## Prerequisites & Installation

### 1. Environment Setup
Python 3.10 or higher is required. A virtual environment is strongly recommended:

```bash
# Clone the repository
git clone https://github.com/MithilSave/NW_Longitudinal-Neurodegeneration-Progression-Modeling.git
cd NW_Longitudinal-Neurodegeneration-Progression-Modeling

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Package & Dependencies
Install dependencies and install the local package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## How to Run the Pipeline

### 1. Quick Start Execution (CLI)
To run the full end-to-end pipeline (Phase 3 NDM, Phase 4 Deficits, Phase 5 Deep Learning Suite, Phase 6 Digital Twin Rollout, and Phase 7 Node Attribution) using preprocessed regional volumes:

```bash
python -m oasis2
```

### 2. Custom Execution Flags
You can specify custom paths and hyperparameters via command-line arguments:

```bash
python -m oasis2 \
  --demo-csv data/raw/oasis_longitudinal_demographics.csv \
  --regional-volumes-csv outputs/regional_volumes.csv \
  --outputs-dir outputs/ \
  --dl-epochs 50
```

#### Available CLI Options:
- `--demo-csv PATH`: Path to demographics spreadsheet (`.csv` or `.xlsx`). Defaults to auto-discovery under `data/raw/`.
- `--regional-volumes-csv PATH`: Path to 56-region volumetric CSV. Defaults to `outputs/regional_volumes.csv`.
- `--outputs-dir PATH`: Directory where model outputs, metrics, and figures are saved. Defaults to `outputs/`.
- `--dl-epochs INT`: Training epochs for ST-GNN-ODE, ND-VAE, and TADM (default: `50`).

### 3. Running the Test Suite
The test suite validates data loading, graph Laplacian math, NDM closed-form solver, deficit transformations, deep learning forward/backward passes, and digital twin rollouts using synthetic test fixtures (no external downloads required):

```bash
pytest tests/
```

All 66 test modules should pass in ~12 seconds.

### 4. Python API Usage
The package can also be scripted programmatically:

```python
import pandas as pd
from oasis2.data.ingest import load_demographic_csv
from oasis2.pipeline import run_pipeline

# 1. Load clinical demographics and parcellations
demo_df = load_demographic_csv("data/raw/oasis_longitudinal_demographics.csv")
regional_volumes = pd.read_csv("outputs/regional_volumes.csv")

# 2. Run end-to-end pipeline
results = run_pipeline(
    demo_df=demo_df,
    regional_volumes=regional_volumes,
    outputs_dir="outputs/",
    dl_epochs=50
)

# 3. Access model evaluation metrics and rollout predictions
print("NDM Pearson correlation:", results["phase3"]["validation"]["pearson_r"])
print("Benchmark comparison:\n", results["benchmark"])
```

---

## How to Run & View the 3D WebGL Digital Twin Simulation

The interactive 3D digital twin is packaged as a **completely self-contained, standalone WebGL application** (`outputs/digital_twin_3d_simulation.html`). It requires no backend server, node installation, or external plugins.

### 1. Launching the Simulation

#### Option A: Direct Open in Browser
Double-click `outputs/digital_twin_3d_simulation.html` in your file explorer, or execute in terminal:

```powershell
# Windows PowerShell
Start-Process outputs/digital_twin_3d_simulation.html
```

```bash
# macOS
open outputs/digital_twin_3d_simulation.html

# Linux
xdg-open outputs/digital_twin_3d_simulation.html
```

#### Option B: Local HTTP Server (Recommended)
If your browser restricts local file access (`file:///` protocol) for embedded Web Workers:

```bash
python -m http.server 8000
```
Navigate to: [http://localhost:8000/outputs/digital_twin_3d_simulation.html](http://localhost:8000/outputs/digital_twin_3d_simulation.html)

---

### 2. Rebuilding the Simulation from Model Checkpoints

If you run the pipeline with new subjects or updated deep learning weights, regenerate the standalone simulation HTML by running:

```bash
python scripts/build_3d_simulation.py
```

This compiles:
- MNI stereotaxic 3D coordinates for all 56 Desikan-Killiany cortical parcels and subcortical nuclei.
- Structural connectome tract lines and weight matrices.
- Longitudinal patient baseline states ($W$-scores, CDR, MMSE, Age).
- Forward-simulated 5-year progression trajectories and 11-domain cognitive deficit vectors.

Output generated: `outputs/digital_twin_3d_simulation.html` (~117 KB).

---

### 3. Interactive Simulation Controls

| Interface Element | Functionality |
| :--- | :--- |
| **Cohort Selector** | Switch between real representative patient trajectories: `Demented (OAS2_0002)`, `Converted (OAS2_0031)`, or `Non-Demented (OAS2_0001)`. |
| **Year Scrub Slider** | Continuous progression scrubbing from **Year 0.0 (Baseline)** to **Year 5.0 (Forward Projection)**. |
| **Play / Pause / Loop** | Auto-play continuous disease progression animation at 1× speed. |
| **3D Brain Viewport** | Left-click + drag to rotate in 3D orbit; scroll wheel to zoom; right-click to pan. Hover over any parcel to inspect anatomical name, MNI coordinate, and real-time $W$-score. |
| **Camera Snap Presets** | Instant anatomical alignment: `Superior (Top)`, `Anterior (Front)`, `Left Lateral`, or `Isometric`. |
| **Cognitive Radar Chart** | Live Chart.js radar tracking 11 clinical domains simultaneously as neurodegeneration spreads. |
| **Tract Lines Toggle** | Show/hide underlying structural connectome axonal edges connecting cortical parcels. |
| **Auto-Rotate Toggle** | Enable continuous anatomical orbital turntable rotation. |

---

## Repository Structure

```
├── README.md                                      # Project overview, installation, and run instructions
├── functional_design_document.md                  # Executive design specification
├── implementation.md                              # Detailed mathematical & biophysical derivations
├── model1_ST_GNN_ODE.md                           # Continuous Spatio-Temporal Graph ODE docs
├── model2_ND_VAE.md                               # Physics-Constrained VAE docs
├── model3_TADM.md                                 # Trajectory-Aware Diffusion Model docs
├── model_implementation_plan.md                   # Model architecture implementation reference
├── pyproject.toml                                 # Python packaging & test configurations
├── requirements.txt                               # Pinned environment dependencies
├── docs/
│   ├── FUNCTIONAL_DESIGN_DOCUMENT.md             # Architectural reference with flowchart
│   └── superpowers/                               # Conversion specs and design history
├── scripts/
│   └── build_3d_simulation.py                     # Standalone 3D WebGL simulator compiler
├── src/oasis2/                                    # Modular Python package
│   ├── cli.py                                     # Command-line entry point (`python -m oasis2`)
│   ├── pipeline.py                                # Master pipeline driver
│   ├── connectome.py                              # Structural connectome & graph Laplacian math
│   ├── ndm.py                                     # Network Diffusion Model closed-form solver
│   ├── deficits.py                                # 11-domain cognitive impairment mapping
│   ├── digital_twin.py                            # Autoregressive rollout & 95% MC bands
│   ├── interpretability.py                        # Integrated Gradients node attribution
│   ├── data/                                      # Parcellation ingestion & dataset builders
│   └── models/                                    # ST-GNN-ODE, ND-VAE, TADM, and benchmark suite
├── tests/                                         # Comprehensive 66-test pytest suite
├── data/
│   ├── raw/                                       # OASIS-2 longitudinal clinical spreadsheets
│   └── processed/
│       └── template_connectome.npy                # 56x56 structural reference connectome
└── outputs/                                       # Generated artifacts
    ├── digital_twin_3d_simulation.html            # Standalone interactive 3D WebGL simulation
    ├── regional_volumes.csv                       # SynthSeg 56 regional brain volumes
    ├── phase3/                                    # W-scores, fitted betas, Laplacian eigenvalues
    ├── phase4/                                    # 5-year atrophy forecasts & cognitive deficit scores
    └── benchmark/                                 # Cross-model validation metrics
```

---

## Technical Documentation Index

Detailed specifications for each sub-system are documented in the repository:
- **Architecture & System Pipeline:** [docs/FUNCTIONAL_DESIGN_DOCUMENT.md](docs/FUNCTIONAL_DESIGN_DOCUMENT.md)
- **Mathematical Formulations:** [implementation.md](implementation.md)
- **Model 1 (ST-GNN-ODE):** [model1_ST_GNN_ODE.md](model1_ST_GNN_ODE.md)
- **Model 2 (ND-VAE):** [model2_ND_VAE.md](model2_ND_VAE.md)
- **Model 3 (TADM):** [model3_TADM.md](model3_TADM.md)
- **Implementation Plan:** [model_implementation_plan.md](model_implementation_plan.md)

---

## Key Methodology & Leakage Prevention

- **Subject-Level Stratification:** Cross-validation strictly partitions data by Subject ID using `GroupKFold`. As demonstrated by Yagis et al. (2021), random slice- or visit-level splitting leaks subject anatomy into test folds, artificially inflating accuracy by up to 30%.
- **Age-Normalized Residual Volumetrics ($W$-Scores):** Evaluating Alzheimer's atrophy using raw volumes or simple $z$-scores conflates normal age-related brain shrinkage with pathological neurodegeneration. Volumes are standardized against Ordinary Least Squares (OLS) regressions fitted exclusively on non-demented healthy controls.
- **Biophysical Template Connectome:** Following the validated framework of Raj et al. (2012, 2015), pathology propagation is modeled as heat-kernel diffusion over a reference structural connectome Laplacian ($L = D - A$).
- **Realistic Performance Metrics:** Subject-level longitudinal progression prediction accuracy of 60–75% aligns with peer-reviewed literature for cohorts of ~150 subjects; unrealistically high scores (>90%) are indicative of cross-sectional label leakage.

---

## References

1. **Raj, A., Kuceyeski, A., & Weiner, M.** (2012). A network diffusion model of disease progression in dementia. *Neuron*, 73(6), 1204–1215.
2. **Raj, A., et al.** (2015). Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. *Cell Reports*, 10(3), 359–369.
3. **Billot, B., et al.** (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Medical Image Analysis*, 86, 102789.
4. **Marcus, D. S., et al.** (2010). Open Access Series of Imaging Studies (OASIS-2): Longitudinal MRI data in nondemented and demented older adults. *Journal of Cognitive Neuroscience*, 22(12), 2677–2684.
5. **Yagis, E., et al.** (2021). Effect of data leakage in brain MRI classification using 2D convolutional neural networks. *Scientific Reports*, 11, 22544.
6. **Chen, R. T. Q., et al.** (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

---

## License

This repository uses the [OASIS-2 dataset](https://www.oasis-brains.org/), available under its respective Data Use Agreement. Core modeling and simulation code is provided for academic and clinical research reference.
