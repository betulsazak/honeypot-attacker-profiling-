<div align="center">

# 🍯 Honeypot Attacker Profiling

### Spatiotemporal Analysis of Honeypot Attack Data & ML-Based Attacker Behavior Classification

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-006600)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

*Graduation Thesis — Biruni University, Department of Computer Engineering, Spring 2026*

**[Overview](#-overview) · [Key Results](#-key-results) · [Pipeline](#-pipeline) · [Setup](#%EF%B8%8F-setup) · [Usage](#-usage) · [Architecture](#-lstm-architecture) · [License](#-license)**

</div>

---

## 📌 Overview

This project applies **traditional machine learning** and **sequence-based deep learning** to large-scale honeypot log data to profile attacker behaviors, classify them, and predict threat levels at an early stage.

Two honeypot datasets from **different network layers** were independently analyzed and compared within a unified framework:

<table>
<tr>
<td align="center"><b>🖥️ Cowrie SSH/Telnet</b><br><sub>Application Layer</sub><br><code>~83.7M rows</code><br><sub>May 2019 – Feb 2020</sub><br><a href="https://zenodo.org/record/4049991">Download</a></td>
<td align="center"><b>🌐 CTU-Hornet 65 Niner</b><br><sub>Network Layer</sub><br><code>~12.5M rows</code><br><sub>Apr – Jul 2024</sub><br><a href="https://zenodo.org/records/13382500">Download</a></td>
</tr>
</table>

---

## 🏆 Key Results

<table>
<tr>
<td width="50%">

### LSTM vs XGBoost

| Scenario | LSTM | XGBoost | Δ |
|:---------|:----:|:-------:|:-:|
| Cowrie Binary | 0.916 | **0.920** | −0.4 |
| **CTU Binary** | **0.979** | 0.890 | **+9.0** |
| Cowrie Early Warning | 0.713 | **0.826** | −11.3 |
| CTU Early Warning | **0.936** | 0.933 | +0.3 |

</td>
<td width="50%">

### Highlights

🔬 **LSTM beat XGBoost by 9 points** on CTU network flow data — proving temporal dynamics carry critical information

📊 **K-Means** profiled 43,513 IPs into Bot (59.5%) and Manual (40.5%) clusters using 4-metric consensus

⚡ **Early warning** achieves reliable prediction with just **3–5 initial attempts**

🧠 **Bi-LSTM** underperformed standard LSTM due to early overfitting in all scenarios

</td>
</tr>
</table>

---

## 🔄 Pipeline

```
                        ┌─────────────────────┐
                        │     Raw Honeypot     │
                        │        Logs          │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼                              ▼
          ┌─────────────────┐            ┌─────────────────┐
          │  Cowrie SSH/Tel  │            │   CTU-Hornet    │
          │  Application     │            │   Network       │
          │  ~83.7M rows     │            │   ~12.5M rows   │
          └────────┬────────┘            └────────┬────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
                   ┌──────────────────────────┐
                   │   Preprocessing &        │
                   │   Exploratory Analysis   │
                   └─────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
   ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐
   │  Unsupervised  │  │   Supervised    │  │    Deep       │
   │  K-Means       │  │   RF, XGBoost   │  │   Learning   │
   │  18 features   │  │   SVM, LR       │  │  LSTM/BiLSTM │
   │  k=2 consensus │  │   Bot/Manual    │  │  Sequence    │
   └───────┬────────┘  └────────┬────────┘  └──────┬───────┘
           │                    │                   │
           └────────────────────┼───────────────────┘
                                ▼
                   ┌──────────────────────────┐
                   │     Early Warning        │
                   │  First 30% interactions  │
                   │  3-class threat level    │
                   │  Window: N=3,5,10,20,50  │
                   └──────────────────────────┘
```

---

## 📊 Detailed Results

<details>
<summary><b>Traditional ML Baseline (Weighted F1)</b></summary>
<br>

| Scenario | RF | XGBoost | SVM | LR |
|:---------|:---:|:-------:|:---:|:---:|
| Cowrie Binary | 0.918 | **0.920** | 0.901 | 0.841 |
| CTU Binary | 0.888 | **0.890** | 0.872 | 0.568 |
| Cowrie Early Warning | 0.824 | **0.826** | 0.780 | 0.707 |
| CTU Early Warning | 0.932 | **0.933** | 0.906 | 0.903 |

</details>

<details>
<summary><b>Sequence LSTM Full Results</b></summary>
<br>

| Scenario | Model | F1 | Accuracy | Precision | Recall | Epochs |
|:---------|:-----:|:--:|:--------:|:---------:|:------:|:------:|
| Cowrie W6 | LSTM | 0.916 | 0.920 | 0.914 | 0.920 | 49 |
| Cowrie W6 | Bi-LSTM | 0.875 | 0.871 | 0.881 | 0.871 | 12 |
| CTU W6 | LSTM | **0.979** | 0.979 | 0.979 | 0.979 | 50 |
| CTU W6 | Bi-LSTM | 0.934 | 0.934 | 0.934 | 0.934 | 12 |
| Cowrie W7 | LSTM | 0.670 | 0.722 | 0.729 | 0.722 | 50 |
| Cowrie W7 | Bi-LSTM | 0.641 | 0.689 | 0.654 | 0.689 | 12 |
| CTU W7 | LSTM | 0.936 | 0.935 | 0.939 | 0.935 | 50 |
| CTU W7 | Bi-LSTM | 0.849 | 0.858 | 0.845 | 0.858 | 12 |

</details>

<details>
<summary><b>K-Means Clustering</b></summary>
<br>

**Cluster Distribution:**

| Profile | IPs | % of IPs | % of Attempts | Avg Attempts | Success Rate |
|:--------|:---:|:--------:|:-------------:|:------------:|:------------:|
| 🤖 Bot | 25,895 | 59.5% | 86.2% | 951.7 | 3.5% |
| 👤 Manual | 17,618 | 40.5% | 13.8% | 224.8 | 12.3% |

**Optimal k Selection (4-Metric Consensus):**

| k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|:-:|:------------:|:----------------:|:-------------------:|
| **2** | **0.299** | 1.432 | **17,474** |
| 3 | 0.255 | 1.469 | 14,907 |
| 4 | 0.274 | 1.480 | 12,655 |
| 7 | 0.249 | **1.287** | 10,175 |

</details>

<details>
<summary><b>SEQ_LEN Ablation Test</b></summary>
<br>

| Model | SEQ_LEN=20 | SEQ_LEN=50 | Δ |
|:-----:|:----------:|:----------:|:-:|
| LSTM | 0.670 | 0.713 | +4.3 |
| Bi-LSTM | 0.641 | 0.711 | +7.0 |

Longer sequences help — especially for Bi-LSTM where bidirectional context needs sufficient length to become meaningful.

</details>

---

## 🧠 LSTM Architecture

```
┌──────────────────────────────────────────┐
│           Input (SEQ_LEN × features)     │
├──────────────────────────────────────────┤
│  Masking (ignore zero-padding)           │
├──────────────────────────────────────────┤
│  LSTM (64 units, return_sequences=True)  │
│  Dropout (0.3)                           │
├──────────────────────────────────────────┤
│  LSTM (32 units)                         │
│  Dropout (0.3)                           │
├──────────────────────────────────────────┤
│  Dense (32, ReLU)                        │
│  Dropout (0.2)                           │
├──────────────────────────────────────────┤
│  Dense (softmax / sigmoid)               │
└──────────────────────────────────────────┘
```

**Per-Event Features:**

| | Cowrie (7 features) | CTU-Hornet (8 features) |
|:--|:--|:--|
| ⏱️ Temporal | delta_log, hour | delta_log, hour |
| 📝 Content | user_len, pass_len, entropy, success | duration_log, orig_bytes_log, resp_bytes_log |
| 🏷️ Categorical | event_code | dst_port, proto, conn_state |

**Training Config:** Adam optimizer · LR: 1e-3 (LSTM), 5e-4 (Bi-LSTM) · Early stopping: patience=12 · 20% validation split · CPU-only

---

## 📁 Project Structure

```
honeypot-attacker-profiling/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 scripts/
│   ├── week5_behavioral_clustering.py       # K-Means + 18 features + bot scoring
│   ├── week6_supervised_classification.py    # RF, XGBoost, SVM, LR
│   ├── week_lstm_sequence.py                # Sequence LSTM/Bi-LSTM (SEQ_LEN=20)
│   └── w7_cowrie_seq50.py                   # Ablation test (SEQ_LEN=50)
│
├── 📂 output/
│   ├── week5/
│   │   ├── figures/                         # Clustering visualizations
│   │   └── data/                            # Cluster metrics, bot scores
│   ├── lstm_seq/
│   │   ├── figures/                         # Confusion matrices, training curves
│   │   └── data/                            # Summary CSVs
│   └── lstm_seq_w7cowrie_seq50/             # Ablation results
│
└── 📂 docs/
    └── thesis_summary.md
```

---

## ⚙️ Setup

### Requirements

| | Minimum | Recommended |
|:--|:--|:--|
| Python | 3.10+ | 3.12 |
| RAM | 8 GB | 16 GB |
| GPU | Not required | Optional (speeds up LSTM training) |
| OS | Windows (WSL), Linux, macOS | Ubuntu 22.04+ (WSL or native) |

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/honeypot-attacker-profiling.git
cd honeypot-attacker-profiling

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / WSL / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.15.0
scikit-learn>=1.4.0
xgboost>=2.0.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
ijson>=3.2.0
geoip2>=4.8.0
```

---

## 🚀 Usage

```bash
# Stage 2 — K-Means clustering with 18 behavioral features
python scripts/week5_behavioral_clustering.py

# Stage 3 — Supervised classification (RF, XGBoost, SVM, LR)
python scripts/week6_supervised_classification.py

# Stage 5 — Sequence LSTM & Bi-LSTM (SEQ_LEN=20)
python scripts/week_lstm_sequence.py

# Ablation — W7 Cowrie with SEQ_LEN=50
python scripts/w7_cowrie_seq50.py
```

> **Note:** Raw datasets are not included in this repository. Download them from the links in the [Overview](#-overview) section and place them in the project directory before running any scripts.

---

## 🔐 Data Privacy

| Dataset | IP Handling | Status |
|:--------|:-----------|:-------|
| Cowrie | SHA-256 hashed | ✅ Safe to share derived features |
| CTU-Hornet | Real IPs | ⚠️ Raw data excluded from repo |

Only derived feature CSVs (without IP columns) are shared in the `output/` directory.

---

## 🎯 Task-Specific Model Recommendation

| Task | Model | Why |
|:-----|:-----:|:----|
| Cowrie binary classification | **XGBoost** | Equivalent to LSTM, much faster inference |
| CTU binary classification | **LSTM** | +9 points — temporal dynamics are critical |
| Cowrie early warning | **XGBoost** | Aggregate features match task definition |
| CTU early warning | **Either** | Equivalent performance |

> **General guidance:** XGBoost for production (speed + interpretability), LSTM for network-layer temporal analysis.

---

## ⚠️ Limitations

- Datasets from different time periods — same attacker cannot be tracked across layers
- Labels are rule-based (threshold-derived), not expert-validated
- CPU-only training limited model size and sequence length exploration
- SEQ_LEN was not optimized via systematic hyperparameter search
- Class imbalance affects minority class performance

## 🔮 Future Work

- **T-Pot** integrated honeypot for simultaneous multi-layer data collection
- **Transformer / Attention** based sequence models
- **Real-time SOC integration** via Apache Kafka or Flink
- **SMOTE / ADASYN** for class imbalance mitigation
- **Active learning** with expert-in-the-loop labeling

---

<div align="center">

## 👤 Author

**Betül Sazak**
Biruni University · Computer Engineering

Advisor: **Dr. Özgür Koray Şahingöz**

---

## 📄 License

Code is released under the **MIT License**.
Datasets are subject to their original source licenses.

</div>
