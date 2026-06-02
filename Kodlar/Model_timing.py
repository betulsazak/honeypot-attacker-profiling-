"""
Model Süre Ölçümü — Hafta 6 ve Hafta 7
========================================
Her model için eğitim süresi ve tahmin süresi ölçer.
Cowrie ve CTU-Hornet ayrı ayrı.
Mevcut etiketli CSV dosyalarını kullanır, parquet okumaz.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/mnt/c/TEZ_PROJEEE/output"
RS = 42


def measure_models(X_train, X_test, y_train, y_test, dataset_name, phase_name):
    """Her model için eğitim ve tahmin süresini ölçer."""
    print(f"\n  {phase_name} — {dataset_name}")
    print(f"  {'─'*55}")
    print(f"  {'Model':<25} {'Eğitim(s)':>10} {'Tahmin(s)':>10} {'F1':>8} {'Acc':>8}")
    print(f"  {'─'*55}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # SVM örneklem
    svm_n = min(10_000, len(X_train_s))
    if svm_n < len(X_train_s):
        idx = np.random.default_rng(RS).choice(len(X_train_s), svm_n, replace=False)
        X_svm, y_svm = X_train_s[idx], y_train[idx]
    else:
        X_svm, y_svm = X_train_s, y_train

    model_configs = [
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RS, n_jobs=-1), X_train, X_test, y_train),
        ("XGBoost", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric="logloss", use_label_encoder=False, n_jobs=-1) if HAS_XGB else None, X_train, X_test, y_train),
        ("SVM", SVC(C=10, kernel="rbf", gamma="scale", random_state=RS), X_svm, X_test_s, y_svm),
        ("Logistic Regression", LogisticRegression(C=1, max_iter=1000, random_state=RS, n_jobs=-1), X_train_s, X_test_s, y_train),
    ]

    for name, model, X_tr, X_te, y_tr in model_configs:
        if model is None:
            continue

        # Eğitim süresi
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        # Tahmin süresi
        t0 = time.time()
        y_pred = model.predict(X_te)
        pred_time = time.time() - t0

        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)

        print(f"  {name:<25} {train_time:>9.3f}s {pred_time:>9.3f}s {f1:>7.4f} {acc:>7.4f}")

        results.append({
            "Phase": phase_name,
            "Dataset": dataset_name,
            "Model": name,
            "Train_Time_s": round(train_time, 3),
            "Predict_Time_s": round(pred_time, 3),
            "F1_Score": round(f1, 4),
            "Accuracy": round(acc, 4),
            "Train_Size": len(X_tr),
            "Test_Size": len(X_te),
        })

    return results


def load_and_run(csv_path, label_col, exclude_cols, dataset_name, phase_name):
    """CSV yükle, özellik/etiket ayır, model süre ölç."""
    if not os.path.exists(csv_path):
        print(f"  {csv_path} bulunamadı — atlanıyor.")
        return []

    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns
                 if c not in exclude_cols and not c.startswith("_")
                 and df[c].dtype in ("float64", "int64", "int32", "float32")]

    X = df[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
    for c in feat_cols:
        if X[c].max() > 100:
            X[c] = np.log1p(X[c])

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y)

    print(f"\n  Veri: {len(df):,} satır, {len(feat_cols)} özellik, sınıflar: {list(le.classes_)}")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    return measure_models(X_train, X_test, y_train, y_test, dataset_name, phase_name)


if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  MODEL SÜRE ÖLÇÜMÜ — Hafta 6 & 7                       ║")
    print("╚" + "═" * 58 + "╝")

    all_results = []

    # ═══ HAFTA 6 — Cowrie ═══
    print("\n" + "=" * 60)
    print("HAFTA 6 — DENETİMLİ SINIFLANDIRMA")
    print("=" * 60)

    r = load_and_run(
        os.path.join(OUTPUT_DIR, "week6", "data", "cowrie_labeled.csv"),
        label_col="label",
        exclude_cols={"src_ip", "label", "manuel_score", "bot_score", "rule_label"},
        dataset_name="Cowrie",
        phase_name="Hafta 6"
    )
    all_results.extend(r)

    r = load_and_run(
        os.path.join(OUTPUT_DIR, "week6", "data", "ctu_labeled.csv"),
        label_col="label",
        exclude_cols={"src_ip", "label", "manuel_score", "bot_score", "rule_label"},
        dataset_name="CTU-Hornet",
        phase_name="Hafta 6"
    )
    all_results.extend(r)

    # ═══ HAFTA 7 — Cowrie Erken Tespit ═══
    print("\n" + "=" * 60)
    print("HAFTA 7 — ERKEN UYARI SİSTEMİ")
    print("=" * 60)

    r = load_and_run(
        os.path.join(OUTPUT_DIR, "week7", "data", "cowrie_early_features.csv"),
        label_col="threat_label",
        exclude_cols={"src_ip", "threat_label", "manuel_score", "bot_score"},
        dataset_name="Cowrie",
        phase_name="Hafta 7"
    )
    all_results.extend(r)

    r = load_and_run(
        os.path.join(OUTPUT_DIR, "week7", "data", "ctu_early_features.csv"),
        label_col="behavior_label",
        exclude_cols={"src_ip", "behavior_label", "bot_score"},
        dataset_name="CTU-Hornet",
        phase_name="Hafta 7"
    )
    all_results.extend(r)

    # ═══ ÖZET TABLO ═══
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)

    if all_results:
        summary = pd.DataFrame(all_results)
        print(f"\n{summary.to_string(index=False)}")

        save_path = os.path.join(OUTPUT_DIR, "model_timing_results.csv")
        summary.to_csv(save_path, index=False)
        print(f"\n  → {save_path}")