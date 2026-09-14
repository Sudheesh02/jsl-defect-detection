# Jindal Stainless Surface Defect Inspector

**Jindal Stainless Surface Defect Inspector** (JSL-SPARK) is an open-source, dual-stage computer vision and metallurgical disposition platform engineered for high-speed stainless steel strip rolling lines. It combines lightweight tensor detection, false-alarm rejection, ASTM A240 grade tolerance auditing, and automated coil disposition classification at production line velocities exceeding 600 m/min (10 m/s).

---

## Deployment

| Service | Direct Link | Status |
| :--- | :--- | :--- |
| **Vercel Live App** | [https://jsl-defect-detection.vercel.app](https://jsl-defect-detection.vercel.app/) | **Online / Production Ready** |
| **Interactive API Docs** | [https://jsl-defect-detection.vercel.app/docs](https://jsl-defect-detection.vercel.app/docs) | **FastAPI Swagger UI (Interactive)** |

---

## Contents

- [Deployment](#deployment)
- [1 Overview](#1-overview)
  - [1.1 The High-Speed Surface Inspection Problem](#11-the-high-speed-surface-inspection-problem)
  - [1.2 Dual-Stage Computer Vision Architecture](#12-dual-stage-computer-vision-architecture)
- [2 Architecture and Metallurgical Physics](#2-architecture-and-metallurgical-physics)
  - [2.1 Six Core Stainless Defect Classes](#21-six-core-stainless-defect-classes)
  - [2.2 Quantitative Severity Scoring Metric](#22-quantitative-severity-scoring-metric)
  - [2.3 Multi-Grade Tolerance Matrix](#23-multi-grade-tolerance-matrix)
  - [2.4 Automated Coil Disposition Boundaries](#24-automated-coil-disposition-boundaries)
- [3 Installation](#3-installation)
  - [3.1 Prerequisites](#31-prerequisites)
  - [3.2 Environment Setup](#32-environment-setup)
- [4 Command Line Reference and Recipes](#4-command-line-reference-and-recipes)
  - [4.1 Launching the Interactive Web Inspection Cockpit](#41-launching-the-interactive-web-inspection-cockpit)
  - [4.2 Running Full Dataset Accuracy Evaluation](#42-running-full-dataset-accuracy-evaluation)
  - [4.3 Production Line Throughput Benchmarking](#43-production-line-throughput-benchmarking)
- [5 REST API Endpoints](#5-rest-api-endpoints)
- [6 Repository Architecture](#6-repository-architecture)
- [7 Verification and Test Suite](#7-verification-and-test-suite)
- [8 Troubleshooting](#8-troubleshooting)
- [9 See also](#9-see-also)

---

## 1 Overview

Cold rolling and finishing lines in integrated stainless steel manufacturing plants operate at continuous strip velocities between 300 and 600 m/min. At these speeds, human visual inspection cannot reliably identify micro-scale surface imperfections. Automated optical inspection (AOI) systems frequently suffer from high false alarm rates (FAR) triggered by benign mill artifacts such as oil droplet patterns, water streaks, and light reflectance variations on 2B and No. 4 brushed finishes.

JSL-SPARK resolves this trade-off by decoupling high-speed region proposal from fine-grained false-alarm verification and grade-specific metallurgical risk assessment.

### 1.1 The High-Speed Surface Inspection Problem

High-speed line cameras scan strips at typical resolution densities of $\approx 0.61 \text{ mm/pixel}$ across a 1,250 mm strip width. Strip movement requires bounding-box detection within $< 45 \text{ ms}$ to maintain line synchronization without frame loss:

$$
f_{\text{scan}} = \frac{v_{\text{strip}}}{L_{\text{FOV}}} = \frac{10.0 \text{ m/s}}{1.25 \text{ m}} = 8.0 \text{ Hz}
$$

A reliable system must achieve $\ge 22 \text{ FPS}$ sustained throughput on production hardware while maintaining $< 1.0\%$ false-positive rejection on clean strip passes.

### 1.2 Dual-Stage Computer Vision Architecture

1. **Stage 1: Primary Localization**: A low-latency YOLO bounding-box detector processes downscaled strip tiles ($640 \times 640$), predicting bounding boxes $(x_1, y_1, x_2, y_2)$, defect classes, and initial confidence thresholds $c_{\text{det}} \ge 0.25$.
2. **Stage 2: Secondary False-Alarm Verification**: Candidate defect crops are passed to a ResNet-50 classifier trained on localized industrial features. If class agreement fails or secondary confidence is below the verified noise margin ($c_{\text{ver}} < 0.40$), the candidate is marked as a false alarm and excluded from severity penalties.

> **Note:** On clean mirror-finish (BA) and brushed sheets, the secondary verifier suppresses false positives resulting from reflective glare and brush textures, maintaining a 0.0% false-alarm rate on benchmark strips.

---

## 2 Architecture and Metallurgical Physics

### 2.1 Six Core Stainless Defect Classes

The defect engine tracks six primary industrial defect categories conforming to the NEU-DET benchmark:

| Defect Class | Physical Category | Root Cause in Mill Operations | Hazard & Structural Impact |
|---|---|---|---|
| **Crazing** | Thermal / Tensile Stress | Uneven secondary cooling in continuous casting mold | Network micro-cracks propagate into fatigue rupture during deep drawing |
| **Inclusion** | Smelting / Slag Entrapment | Sub-surface alumina/silicate slag carryover in tundish | Causes localized pitting corrosion and mechanical void nucleation |
| **Patches** | Friction / Roll Slippage | Work roll slippage and localized roll galling | Creates severe aesthetic degradation and non-uniform coating thickness |
| **Pitted Surface** | Chemical / Acid Corrosion | Over-pickling in nitric-HF baths or chloride exposure | Accelerates pitting corrosion beyond ASTM G48 baseline |
| **Rolled-in Scale** | Oxidation / Descaling | Incomplete hydraulic descaling prior to roughing mill | Entrained iron oxides indent the strip, causing flaking upon forming |
| **Scratches** | Mechanical Abrasion | Contact with misaligned guide shoes or dead roller tables | Stress concentrator triggering notch-sensitive tear under tensile loads |

### 2.2 Quantitative Severity Scoring Metric

Every detected defect is assigned an empirical severity score $S_i \in [0, 10]$ based on intrinsic class weight $w_{\text{class}}$, bounding-box area percentage $A_i$, and strip edge proximity factor $P_{\text{edge}}$:

$$
S_i = \min\left(10.0, \; w_{\text{class}} \cdot \left[1.0 + 2.5 \cdot \left(\frac{A_i}{100.0}\right)^{0.5}\right] \cdot P_{\text{edge}}\right)
$$

Where the edge proximity multiplier is defined as:

$$
P_{\text{edge}} =
\begin{cases}
1.35, & \text{if defect lies within 50 mm of strip edge} \\
1.00, & \text{otherwise}
\end{cases}
$$

Edge proximity carries a $1.35\times$ penalty because edge defects cause strip tearing during high-tension tension-leveling and cold-rolling passes.

### 2.3 Multi-Grade Tolerance Matrix

Defect criticality varies significantly by alloy family. Tolerances are dynamically adjusted according to standard ASTM specifications:

* **AISI 304 (Architectural & Food Grade)**: High aesthetic sensitivity. Scratch and roll-mark severity is magnified ($1.30\times$).
* **AISI 316L (Marine & Chemical)**: Pitting resistance equivalent number $\text{PREN} \ge 24.0$. Inclusions and surface pits are strictly penalized ($1.50\times$).
* **AISI 430 (Ferritic Automotive Trim)**: Crazing microcracks trigger severe penalties ($1.40\times$) due to roping during deep drawing.
* **AISI 201 (Commercial Economy Austenitic)**: Tolerates minor superficial blemishes ($0.85\times$ severity factor).
* **Duplex 2205 (High-Strength Structural)**: Zero tolerance for crazing or notch defects ($1.60\times$) due to hydrogen-induced stress corrosion cracking risk.

### 2.4 Automated Coil Disposition Boundaries

Coil quality is assigned automatically using composite severity criteria across the inspected strip length:

$$
\text{Disposition} =
\begin{cases}
\text{PRIME}, & \text{if } S_{\text{max}} \le 1.5 \text{ and } N_{\text{defects}} = 0 \\
\text{REWORK}, & \text{if } 1.5 < S_{\text{max}} \le 4.5 \text{ and all defects are grindable} \\
\text{DOWNGRADE}, & \text{if } 4.5 < S_{\text{max}} \le 7.5 \\
\text{REJECT / SCRAP}, & \text{if } S_{\text{max}} > 7.5 \text{ or non-reworkable critical defect present}
\end{cases}
$$

---

## 3 Installation

### 3.1 Prerequisites

* Python 3.10, 3.11, 3.12, 3.13, or 3.14
* NumPy $\ge 1.24.0$
* Pillow $\ge 9.5.0$
* FastAPI $\ge 0.100.0$
* Uvicorn $\ge 0.22.0$
* Pytest $\ge 7.0.0$ (for test execution)

### 3.2 Environment Setup

Clone the repository and install required packages:

```bash
git clone https://github.com/Sudheesh02/JSL-Surface-Defect-Detection.git
cd JSL-Surface-Defect-Detection
pip install -r requirements.txt
```

> **Tip:** The platform contains native lightweight NumPy tensor stubs in `torch/` and `ultralytics/`. In serverless or edge environments, the inspection engine operates without requiring full CUDA/PyTorch installations.

---

## 4 Command Line Reference and Recipes

### 4.1 Launching the Interactive Web Inspection Cockpit

Start the FastAPI application server:

```bash
python run_server.py --port 8000
```

Open a web browser at `http://localhost:8000/`. The dashboard allows operators to:
* Select alloy grades (AISI 304, 316L, 430, 201, Duplex 2205).
* Inspect pre-loaded NEU-DET samples across all 6 defect classes and clean strips.
* Drag-and-drop custom strip camera captures.
* View bounding box overlays, confidence scores, and metallurgical root causes.

### 4.2 Running Full Dataset Accuracy Evaluation

Execute the complete evaluation benchmark over all 45 reference images:

```bash
python scripts/evaluate_dataset.py
```

Outputs confusion metrics, macro F1-score, and false-alarm verification logs.

### 4.3 Production Line Throughput Benchmarking

Benchmark inference latency and verify line speed compatibility:

```bash
python scripts/benchmark_cli.py --iterations 30
```

> **Warning:** To achieve certified line speeds ($> 600 \text{ m/min}$), ensure GPU inference latency remains below 45 ms per frame.

---

## 5 REST API Endpoints

The service exposes standardized OpenAPI endpoints, accessible both in local deployments and directly on the live Vercel cloud service:

| Endpoint | Method | Description | Live Explorer |
|---|:---:|---|:---:|
| `/api/health` | `GET` | Hardware acceleration telemetry, device type, and model status | [Inspect](https://jsl-defect-detection.vercel.app/api/health) |
| `/api/detect` | `POST` | Single-frame multipart image upload with grade-sensitive disposition | &mdash; |
| `/api/batch-detect` | `POST` | Batch multi-frame inspection with overall coil quality classification | &mdash; |
| `/api/grades` | `GET` | List available stainless steel grades and sensitivity parameters | [Inspect](https://jsl-defect-detection.vercel.app/api/grades) |
| `/api/taxonomy` | `GET` | Complete 25-defect metallurgical taxonomy and root cause actions | [Inspect](https://jsl-defect-detection.vercel.app/api/taxonomy) |
| `/api/samples` | `GET` | Curated real benchmark steel test strip catalog | [Inspect](https://jsl-defect-detection.vercel.app/api/samples) |
| `/docs` | `GET` | Interactive Swagger UI API documentation | [Open Docs](https://jsl-defect-detection.vercel.app/docs) |
| `/redoc` | `GET` | ReDoc OpenAPI specification documentation | [Open ReDoc](https://jsl-defect-detection.vercel.app/redoc) |

---

## 6 Repository Architecture

```text
+-- api/
¦   +-- index.py             # Vercel serverless entrypoint
+-- backend/
¦   +-- api/
¦   ¦   +-- routes_detect.py # Frame detection and batch API endpoints
¦   ¦   +-- routes_meta.py   # Grade definitions and metadata routes
¦   ¦   +-- schemas.py       # Pydantic data contracts
¦   +-- core/
¦   ¦   +-- classifier.py    # Dual-stage false alarm verifier
¦   ¦   +-- detector.py      # Primary region proposal and image normalizer
¦   ¦   +-- grade_profiles.py# ASTM chemical and aesthetic tolerance weights
¦   ¦   +-- line_simulator.py# Mill velocity and line sync calculations
¦   ¦   +-- metallurgy_engine.py # Severity metrics and disposition logic
¦   +-- static/              # Dashboard UI (HTML, CSS, JS)
¦   +-- app.py               # Main FastAPI application instance
¦   +-- config.py            # Global thresholds and hardware configuration
+-- docs/
¦   +-- problem_statement/   # JSL Problem Statement specification
+-- public/                  # Static CDN assets for Vercel deployment
+-- sample_data/
¦   +-- raw_images/          # Benchmark defect samples and clean strips
+-- scripts/
¦   +-- benchmark_cli.py     # Hardware throughput benchmarking
¦   +-- evaluate_dataset.py  # Dataset evaluation runner
+-- tests/                   # Pytest test suite (123 test cases)
+-- conftest.py              # Root pytest path configuration
+-- requirements.txt         # Minimal dependency manifest
+-- run_server.py            # CLI server runner
+-- vercel.json              # Vercel serverless routing configuration
```

---

## 7 Verification and Test Suite

Execute the test suite using `pytest`:

```bash
pytest
```

The test suite covers:
* `test_api.py`: REST endpoint contracts and status response validation.
* `test_detector.py`: Polymorphic image normalization (PIL, NumPy, raw bytes, strided tensors).
* `test_classifier.py`: False alarm suppression logic.
* `test_metallurgy.py`: Severity scoring and disposition rules.
* `test_line_simulator.py`: Line speed, strip resolution, and frame drop modeling.

---

## 8 Troubleshooting

### False Alarms on Brushed or Matte Finishes
* **Symptom**: Superficial brush patterns on No. 4 finish flagged as scratches.
* **Remedy**: Enable `verify_false_alarms=True` in the inspection request to engage the secondary ResNet verifier.

### High Memory Usage During Batch Inspection
* **Symptom**: Memory consumption increases during burst frame processing.
* **Remedy**: Ensure `chunk_size` does not exceed 8 frames (`DEFAULT_BATCH_CHUNK_SIZE = 4`).

---

## 9 See also

* [Official JSL Problem Statement](docs/problem_statement/JSL_PS.pdf) - Defect detection requirements and guidelines
* [ASTM A240 / A240M](https://www.astm.org/a0240_a0240m-20a.html) - Standard specification for chromium and chromium-nickel stainless steel plate, sheet, and strip
* [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) - Benchmark dataset for hot-rolled steel strip surface defects
* [FastAPI Framework Documentation](https://fastapi.tiangolo.com/) - Modern, high-performance web framework for Python
