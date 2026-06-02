"""
CTU-Hornet Pencere Analizi — İlk N akış ile erken tespit performansı
=====================================================================
Hafta 7'deki Cowrie pencere analizinin CTU versiyonu.
N=5, 10, 20, 30, 50 akış ile F1 skorlarını karşılaştırır.
"""

import os
import gc
import math
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/mnt/c/TEZ_PROJEEE/output"
CTU_PATH   = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")
SAVE_DIR   = os.path.join(OUTPUT_DIR, "week7", "figures")
DATA_DIR   = os.path.join(OUTPUT_DIR, "week7", "data")
BATCH_SIZE = 500_000
RS = 42

C_HIGH = "#E74C3C"; C_BLUE = "#2E86AB"


def _get_cols(p):
    s = pq.read_schema(p)
    return [s.field(i).name for i in range(len(s))]


def _entropy(s):
    if not s or len(s) == 0: return 0.0
    f = Counter(s); n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in f.values())


def run_ctu_window_analysis(ctu_path, windows=[5, 10, 20, 30, 50]):
    print(f"  CTU pencere analizi (N={windows})...")
    cols = _get_cols(ctu_path)

    # ── Tüm IP'lerin akış bilgilerini topla (bir kez) ──
    print("    Akış verileri toplanıyor...")
    ip_early_data = defaultdict(list)  # ip → [(dst_port, duration, proto)]
    ip_total_flows = Counter()
    ip_all_ports = defaultdict(set)

    pf = pq.ParquetFile(ctu_path)
    rc = ["src_ip", "dst_port"]
    if "duration" in cols: rc.append("duration")
    if "proto" in cols: rc.append("proto")

    bn = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=rc):
        df = batch.to_pandas()
        for ip, g in df.groupby("src_ip"):
            ip_total_flows[ip] += len(g)
            ip_all_ports[ip].update(g["dst_port"].dropna().unique())

            # Erken akış verileri (max pencere kadar)
            max_n = max(windows)
            if len(ip_early_data[ip]) < max_n:
                remaining = max_n - len(ip_early_data[ip])
                for _, row in g.head(remaining).iterrows():
                    ip_early_data[ip].append({
                        "port": row.get("dst_port"),
                        "dur": row.get("duration", 0) if "duration" in g.columns else 0,
                        "proto": row.get("proto", "") if "proto" in g.columns else "",
                    })
        del df; gc.collect()
        bn += 1
        if bn % 20 == 0: print(f"      batch {bn}...")

    # En az max(windows)+10 akışı olan IP'ler
    min_flows = max(windows) + 10
    active_ips = [ip for ip in ip_total_flows if ip_total_flows[ip] >= min_flows]
    print(f"    {len(active_ips):,} aktif IP (≥{min_flows} akış)")

    # ── Etiketler (tüm akış geçmişinden — sabit) ──
    ip_labels = {}
    for ip in active_ips:
        total = ip_total_flows[ip]
        unique_ports = len(ip_all_ports.get(ip, set()))
        port_scan = unique_ports > 20
        high_volume = total > 500
        targeted = unique_ports <= 5 and total > 10

        if port_scan and high_volume:
            ip_labels[ip] = "Agresif"
        elif port_scan:
            ip_labels[ip] = "Keşif"
        elif targeted:
            ip_labels[ip] = "Hedefli"
        else:
            ip_labels[ip] = "Keşif"

    label_counts = Counter(ip_labels.values())
    print(f"    Etiket dağılımı: {dict(label_counts)}")

    # ── Her pencere boyutu için model eğit ──
    results = {}
    for N in windows:
        print(f"\n    Pencere N={N}...")
        records = []
        for ip in active_ips:
            early = ip_early_data.get(ip, [])[:N]
            if len(early) < N:
                continue

            ports = [e["port"] for e in early if e["port"] is not None]
            durs = [e["dur"] for e in early if e["dur"] is not None and not np.isnan(e["dur"])]
            protos = [e["proto"] for e in early if e["proto"]]

            unique_ports = len(set(ports))
            common_ports = {22, 23, 80, 443, 445, 3389, 8080, 8443}
            common_ratio = sum(1 for p in ports if p in common_ports) / max(len(ports), 1)

            records.append({
                "src_ip": ip,
                "early_unique_ports": unique_ports,
                "early_avg_duration": np.mean(durs) if durs else 0,
                "early_std_duration": np.std(durs) if len(durs) > 1 else 0,
                "early_proto_diversity": len(set(protos)),
                "early_port_entropy": _entropy([str(p) for p in ports]),
                "early_common_port_ratio": common_ratio,
                "label": ip_labels[ip],
            })

        wdf = pd.DataFrame(records)
        feat_cols = ["early_unique_ports", "early_avg_duration", "early_std_duration",
                     "early_proto_diversity", "early_port_entropy", "early_common_port_ratio"]

        X = wdf[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
        for c in feat_cols:
            if X[c].max() > 100: X[c] = np.log1p(X[c])

        le = LabelEncoder()
        y = le.fit_transform(wdf["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RS, stratify=y)

        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RS, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        f1_w = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)

        # Sınıf bazlı F1
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        class_f1 = {cls: report[cls]["f1-score"] for cls in le.classes_}

        results[N] = {"f1": f1_w, "accuracy": acc, "class_f1": class_f1, "le": le}
        print(f"      F1={f1_w:.4f}, Acc={acc:.4f}")
        for cls, f1 in class_f1.items():
            print(f"        {cls}: F1={f1:.4f}")

    del ip_early_data, ip_total_flows, ip_all_ports, ip_labels
    gc.collect()

    return results


def plot_ctu_window(results, save_dir):
    ns = sorted(results.keys())
    f1s = [results[n]["f1"] for n in ns]
    accs = [results[n]["accuracy"] for n in ns]

    # Tüm sınıfların isimlerini al
    all_classes = list(results[ns[0]]["class_f1"].keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sol: Genel F1 ve Accuracy
    ax1.plot(ns, f1s, "o-", color=C_HIGH, linewidth=2, markersize=8, label="F1-Score")
    ax1.plot(ns, accs, "s--", color=C_BLUE, linewidth=2, markersize=8, label="Accuracy")
    for n, f1 in zip(ns, f1s):
        ax1.annotate(f"{f1:.3f}", (n, f1), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color=C_HIGH, fontweight="bold")
    ax1.set_xlabel("Erken Tespit Penceresi (İlk N Akış)")
    ax1.set_ylabel("Skor")
    ax1.set_title("CTU-Hornet — Genel Performans")
    ax1.set_xticks(ns)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Sağ: Sınıf bazlı F1
    colors = {"Agresif": "#E74C3C", "Hedefli": "#2E86AB", "Keşif": "#F18F01"}
    for cls in all_classes:
        cls_f1s = [results[n]["class_f1"].get(cls, 0) for n in ns]
        ax2.plot(ns, cls_f1s, "o-", linewidth=2, markersize=6,
                 label=f"{cls}", color=colors.get(cls, "#999"))
        for n, f1 in zip(ns, cls_f1s):
            ax2.annotate(f"{f1:.2f}", (n, f1), textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color=colors.get(cls, "#999"))

    ax2.set_xlabel("Erken Tespit Penceresi (İlk N Akış)")
    ax2.set_ylabel("F1-Score")
    ax2.set_title("CTU-Hornet — Sınıf Bazlı Performans")
    ax2.set_xticks(ns)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Erken Uyarı — Kaç Akış Sonra Güvenilir Tahmin?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "ctu_early_detection_window.png")
    plt.savefig(p); plt.close(fig)
    print(f"\n    → {p}")


if __name__ == "__main__":
    print("╔" + "═" * 56 + "╗")
    print("║  CTU-HORNET PENCERE ANALİZİ (N=5,10,20,30,50)      ║")
    print("╚" + "═" * 56 + "╝")
    t0 = time.time()

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    results = run_ctu_window_analysis(CTU_PATH, windows=[5, 10, 20, 30, 50])
    plot_ctu_window(results, SAVE_DIR)

    # CSV kaydet
    rows = []
    for n, r in sorted(results.items()):
        row = {"N": n, "F1_weighted": r["f1"], "Accuracy": r["accuracy"]}
        for cls, f1 in r["class_f1"].items():
            row[f"F1_{cls}"] = f1
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "ctu_window_analysis.csv"), index=False)

    print(f"\n  Özet:")
    for n in sorted(results.keys()):
        r = results[n]
        cls_str = " | ".join(f"{cls}={f1:.3f}" for cls, f1 in r["class_f1"].items())
        print(f"    N={n:>2}: F1={r['f1']:.4f} | {cls_str}")

    print(f"\n  Tamamlandı — {time.time()-t0:.0f}s")