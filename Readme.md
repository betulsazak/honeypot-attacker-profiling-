Honeypot Attacker Profiling

Spatiotemporal Analysis of Honeypot Attack Data and ML-Based Attacker Behavior Classification

Graduation thesis project — Biruni University, Department of Computer Engineering, Spring 2026.
This project applies traditional machine learning and sequence-based deep learning to large-scale honeypot logs in order to profile attacker behaviors, classify them, and predict threat levels at an early stage.
Overview
Two honeypot datasets from different network layers were independently analyzed and compared:
DatasetLayerSizePeriodSourceCowrie SSH/TelnetApplication~83.7M rowsMay 2019 – Feb 2020ZenodoCTU-Hornet 65 NinerNetwork~12.5M rowsApr – Jul 2024Zenodo
Pipeline
Raw Logs → Preprocessing → Exploratory Analysis → Feature Engineering
    ↓
    ├── Unsupervised: K-Means Clustering (18 features, 4-metric consensus)
    ├── Supervised: RF, XGBoost, SVM, Logistic Regression
    ├── Deep Learning: Sequence LSTM & Bi-LSTM
    └── Early Warning: First 30% of interactions → 3-class threat prediction
Key Results
Traditional ML (Weighted F1)
ScenarioRFXGBoostSVMLRCowrie Binary0.9180.9200.9010.841CTU Binary0.8880.8900.8720.568Cowrie Early Warning0.8240.8260.7800.707CTU Early Warning0.9320.9330.9060.903
Sequence LSTM vs XGBoost
ScenarioLSTMXGBoostΔWinnerCowrie Binary0.9160.920−0.4XGBoost (marginal)CTU Binary0.9790.890+9.0LSTMCowrie Early Warning (SEQ=50)0.7130.826−11.3XGBoostCTU Early Warning0.9360.933+0.3LSTM (marginal)
K-Means Clustering

Bot profile: 25,895 IPs (59.5%), responsible for 86.2% of all attempts
Manual profile: 17,618 IPs (40.5%), higher success rate and interactive sessions
Optimal k=2 via consensus of Silhouette (0.298), Davies-Bouldin, Calinski-Harabasz, and Inertia

Main Findings

LSTM outperformed XGBoost by 9 points on CTU network flow data — temporal dynamics in sequential connections carry information that aggregate features cannot capture
On Cowrie application-layer data, LSTM and XGBoost performed equivalently — aggregate features already summarize behavior effectively
Early warning system achieves reliable predictions with as few as 3–5 initial attempts
Bi-LSTM underperformed standard LSTM across all scenarios due to early overfitting

Project Structure
honeypot-attacker-profiling/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── week5_behavioral_clustering.py      # K-Means + 18 features + bot scoring
│   ├── week6_supervised_classification.py   # RF, XGBoost, SVM, LR
│   ├── week_lstm_sequence.py               # Sequence LSTM/Bi-LSTM (SEQ_LEN=20)
│   └── w7_cowrie_seq50.py                  # Ablation test (SEQ_LEN=50)
│
├── output/
│   ├── week5/
│   │   ├── figures/                        # Clustering visualizations
│   │   └── data/                           # Cluster metrics, bot scores
│   ├── lstm_seq/
│   │   ├── figures/                        # Confusion matrices, training curves
│   │   └── data/                           # Summary CSVs
│   └── lstm_seq_w7cowrie_seq50/            # Ablation results
│
└── docs/
    └── thesis_summary.md
LSTM Architecture
Input (SEQ_LEN × features)
  → Masking (ignore padding)
  → LSTM(64, return_sequences=True)
  → Dropout(0.3)
  → LSTM(32)
  → Dropout(0.3)
  → Dense(32, ReLU)
  → Dropout(0.2)
  → Dense(softmax / sigmoid)
Per-event features:
Cowrie (7)CTU-Hornet (8)Temporaldelta_log, hour/23delta_log, hour/23Contentuser_len/32, pass_len/32, entropy/8, successduration_log, orig_bytes_log, resp_bytes_logCategoricalevent_code/7dst_port/65535, proto/3, conn_state/12
Training: Adam optimizer (LR=1e-3 LSTM, 5e-4 Bi-LSTM), early stopping (patience=12), 20% validation split, CPU-only.
Setup
Requirements

Python 3.12+
16 GB RAM recommended
CPU is sufficient (GPU optional)

Installation
bashgit clone https://github.com/YOUR_USERNAME/honeypot-attacker-profiling.git
cd honeypot-attacker-profiling

python -m venv .venv
source .venv/bin/activate        # Linux / WSL
# or
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
Dependencies
tensorflow>=2.15.0
scikit-learn>=1.4.0
xgboost>=2.0.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
ijson>=3.2.0
geoip2>=4.8.0
Usage
bash# Stage 2 — K-Means clustering with 18 behavioral features
python scripts/week5_behavioral_clustering.py

# Stage 3 — Supervised classification (RF, XGBoost, SVM, LR)
python scripts/week6_supervised_classification.py

# Stage 5 — Sequence LSTM & Bi-LSTM (SEQ_LEN=20)
python scripts/week_lstm_sequence.py

# Ablation — W7 Cowrie with SEQ_LEN=50
python scripts/w7_cowrie_seq50.py
Note: Raw datasets are not included. Download them from the links in the Overview table and place them in the project directory before running the scripts.
Data Privacy

Cowrie IPs are SHA-256 hashed — no geographic analysis possible
CTU-Hornet contains real IPs — raw data files are excluded from this repository
Only derived feature CSVs (without IP columns) are shared in output/

Limitations

Datasets are from different time periods — same attacker cannot be tracked across layers
Labels are rule-based (threshold-derived), not expert-validated
CPU-only training environment limited model size and sequence length exploration
SEQ_LEN was not optimized via systematic hyperparameter search
Class imbalance affects minority class performance (Manual, Medium, Aggressive)

Future Work

Simultaneous multi-layer data collection using T-Pot integrated honeypot framework
Transformer / Attention-based sequence models
Real-time SOC integration via Apache Kafka or Flink
SMOTE / ADASYN for class imbalance mitigation
Active learning with expert-in-the-loop labeling

Task-Specific Model Recommendation
TaskRecommended ModelRationaleCowrie binary classificationXGBoostEquivalent to LSTM, much fasterCTU binary classificationSequence LSTM+9 points over XGBoostCowrie early warningXGBoostAggregate features match task definitionCTU early warningEitherEquivalent performance
General guidance: Use XGBoost for production (speed + interpretability), LSTM for network-layer temporal analysis.