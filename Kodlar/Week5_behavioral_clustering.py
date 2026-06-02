"""
Hafta 5: Davranışsal Kümeleme — Bot vs Manuel Sınıflandırma (v7 uyumlu)
========================================================================
Cowrie ve CTU-Hornet verilerinden saldırgan davranış özelliklerini çıkarıp
K-Means kümeleme ile bot/manuel/hibrit sınıflandırması yapar.

Pipeline:
  ═══════════════════════════════════════════════════════════
  1. ÖZELLİK MÜHENDİSLİĞİ  (Chunk-chunk, bellek-dostu)
  ═══════════════════════════════════════════════════════════
     Cowrie — IP başına:
       F1.  total_attempts        : Toplam login denemesi
       F2.  unique_usernames      : Benzersiz kullanıcı adı sayısı
       F3.  unique_passwords      : Benzersiz şifre sayısı
       F4.  unique_combos         : Benzersiz user:pass kombinasyonu
       F5.  success_rate          : Başarılı giriş oranı
       F6.  avg_time_between      : Denemeler arası ortalama süre (saniye)
       F7.  session_count         : Toplam oturum sayısı
       F8.  commands_per_session  : Oturum başına ortalama komut sayısı
       F9.  unique_hours          : Aktif olduğu farklı saat sayısı (0-24)
       F10. hour_entropy          : Saat dağılımının entropisi (düzgün=bot)
       F11. avg_password_entropy  : Denenen şifrelerin ortalama entropisi
       F12. avg_password_length   : Denenen şifrelerin ortalama uzunluğu
       F13. password_reuse_ratio  : Şifre tekrar kullanım oranı (attempts/unique)
       F14. duration_mean         : Ortalama oturum süresi
       F15. has_file_download     : Dosya indirme olayı var mı (0/1)

     CTU-Hornet — IP başına:
       F16. flow_count            : Toplam ağ akışı sayısı
       F17. unique_dst_ports      : Hedeflenen benzersiz port sayısı
       F18. unique_dst_ips        : Hedeflenen benzersiz IP sayısı
       F19. avg_flow_duration     : Ortalama akış süresi
       F20. proto_diversity       : Kullanılan protokol çeşitliliği

  ═══════════════════════════════════════════════════════════
  2. ÖN İŞLEME
  ═══════════════════════════════════════════════════════════
     - Log dönüşümü (sağa çarpık dağılımlar için)
     - StandardScaler ile normalizasyon
     - PCA ile boyut indirgeme (görselleştirme için)

  ═══════════════════════════════════════════════════════════
  3. K-MEANS KÜMELEME
  ═══════════════════════════════════════════════════════════
     - Elbow method (k=2..10) ile optimal küme sayısı
     - Silhouette analizi
     - K-Means eğitimi
     - Küme etiketleme: Bot / Manuel / Hibrit

  ═══════════════════════════════════════════════════════════
  4. GÖRSELLEŞTİRME & DEĞERLENDİRME
  ═══════════════════════════════════════════════════════════
     - Elbow grafiği
     - Silhouette grafiği
     - PCA 2D scatter (küme renklendirmeli)
     - Küme profilleri radar chart
     - Özellik önem sıralaması (küme merkezleri farkı)
     - Küme bazlı istatistik tablosu
     - Bot vs Manuel davranış karşılaştırması

  Tüm çıktılar → OUTPUT_DIR/week5/
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
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# YAPILANDIRMA
# =====================================================================

OUTPUT_DIR  = "/mnt/c/TEZ_PROJEEE/output"
WEEK5_DIR   = os.path.join(OUTPUT_DIR, "week5")
WEEK5_FIGS  = os.path.join(WEEK5_DIR, "figures")
WEEK5_DATA  = os.path.join(WEEK5_DIR, "data")

COWRIE_PATH = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")
CTU_PATH    = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")

BATCH_SIZE  = 500_000

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 10, "axes.titlesize": 14,
    "axes.labelsize": 12, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
})

C_BOT    = "#E74C3C"
C_MANUAL = "#2E86AB"
C_HYBRID = "#F18F01"
C_DARK   = "#2C3E50"


# =====================================================================
# YARDIMCI
# =====================================================================

def _get_columns(path):
    schema = pq.read_schema(path)
    return [schema.field(i).name for i in range(len(schema))]

def shannon_entropy(s):
    if not s or len(s) == 0: return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c/length) * math.log2(c/length) for c in freq.values())

def distribution_entropy(counts_dict):
    """Bir dağılımın (saat dağılımı gibi) entropisi."""
    total = sum(counts_dict.values())
    if total == 0: return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counts_dict.values() if c > 0)


# =====================================================================
# 1. ÖZELLİK MÜHENDİSLİĞİ — COWRIE
# =====================================================================

def extract_cowrie_features(cowrie_path):
    """
    Cowrie parquet'ten IP başına 15 davranış özelliği çıkarır.
    İki geçişli: 1) IP başına temel sayımlar 2) Detaylı özellikler
    """
    print("\n  Cowrie özellik mühendisliği...")
    cols = _get_columns(cowrie_path)

    # ── Geçiş 1: IP başına temel sayımlar ──
    print("    Geçiş 1: Temel sayımlar...")
    ip_total_login    = Counter()
    ip_success        = Counter()
    ip_failed         = Counter()
    ip_sessions       = Counter()
    ip_commands        = Counter()
    ip_downloads       = Counter()
    ip_hours           = defaultdict(Counter)  # ip → {hour: count}
    ip_unique_users    = defaultdict(set)
    ip_unique_passes   = defaultdict(set)

    # Zaman damgası bilgileri
    ip_timestamps      = defaultdict(list)  # ip → [timestamp_list] (örneklem)
    ip_durations       = defaultdict(list)

    MAX_TS_PER_IP = 200  # Bellek koruması: IP başına max timestamp

    pf = pq.ParquetFile(cowrie_path)
    read_cols = ["eventid", "src_ip", "hour"]
    if "username" in cols: read_cols.append("username")
    if "password" in cols: read_cols.append("password")
    if "timestamp" in cols: read_cols.append("timestamp")
    if "duration" in cols: read_cols.append("duration")

    batch_num = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        df = batch.to_pandas()

        for eid_pattern, counter in [
            ("login.success", ip_success),
            ("login.failed", ip_failed),
            ("session.connect", ip_sessions),
            ("command.input", ip_commands),
            ("file_download", ip_downloads),
        ]:
            mask = df["eventid"].astype(str).str.contains(eid_pattern, na=False)
            if mask.any():
                counter.update(df.loc[mask, "src_ip"].dropna().value_counts().to_dict())

        # Login toplam
        login_mask = df["eventid"].astype(str).str.contains("login", na=False)
        logins = df[login_mask]
        if len(logins) > 0:
            ip_total_login.update(logins["src_ip"].dropna().value_counts().to_dict())

            # Saat dağılımı
            valid_h = logins[["src_ip", "hour"]].dropna()
            for ip, grp in valid_h.groupby("src_ip"):
                ip_hours[ip].update(grp["hour"].astype(int).value_counts().to_dict())

            # Benzersiz user/pass
            if "username" in logins.columns:
                valid = logins[["src_ip", "username"]].dropna()
                for ip, grp in valid.groupby("src_ip"):
                    ip_unique_users[ip].update(grp["username"].unique())

            if "password" in logins.columns:
                valid = logins[["src_ip", "password"]].dropna()
                for ip, grp in valid.groupby("src_ip"):
                    ip_unique_passes[ip].update(grp["password"].unique())

        # Timestamp (denemeler arası süre için)
        if "timestamp" in df.columns:
            login_ts = df.loc[login_mask, ["src_ip", "timestamp"]].dropna()
            for ip, grp in login_ts.groupby("src_ip"):
                if len(ip_timestamps[ip]) < MAX_TS_PER_IP:
                    ip_timestamps[ip].extend(grp["timestamp"].tolist()[:MAX_TS_PER_IP - len(ip_timestamps[ip])])

        # Duration (oturum süresi)
        if "duration" in df.columns:
            closed_mask = df["eventid"].astype(str).str.contains("session.closed", na=False)
            dur_valid = df.loc[closed_mask, ["src_ip", "duration"]].dropna()
            for ip, grp in dur_valid.groupby("src_ip"):
                ip_durations[ip].extend(grp["duration"].tolist()[:100])

        del df, logins; gc.collect()
        batch_num += 1
        if batch_num % 30 == 0:
            print(f"      batch {batch_num}...")

    # ── Geçiş 2: Şifre entropi (top IP'lerin şifreleri üzerinden) ──
    print("    Şifre entropi hesaplanıyor...")
    ip_pass_entropy = {}
    ip_pass_length  = {}
    for ip, passes in ip_unique_passes.items():
        if passes:
            entropies = [shannon_entropy(str(p)) for p in list(passes)[:500]]
            lengths = [len(str(p)) for p in list(passes)[:500]]
            ip_pass_entropy[ip] = np.mean(entropies)
            ip_pass_length[ip] = np.mean(lengths)

    # ── Özellik DataFrame'i oluştur ──
    print("    Özellik vektörleri oluşturuluyor...")
    all_ips = set(ip_total_login.keys()) | set(ip_sessions.keys())

    # Bellek koruması: en az 5 denemesi olan IP'ler
    active_ips = [ip for ip in all_ips if ip_total_login.get(ip, 0) >= 5]
    print(f"    {len(all_ips):,} IP → {len(active_ips):,} aktif IP (≥5 deneme)")

    records = []
    for ip in active_ips:
        total = ip_total_login.get(ip, 0)
        success = ip_success.get(ip, 0)
        unique_u = len(ip_unique_users.get(ip, set()))
        unique_p = len(ip_unique_passes.get(ip, set()))

        # Denemeler arası süre
        ts_list = sorted(ip_timestamps.get(ip, []))
        if len(ts_list) >= 2:
            diffs = [(ts_list[i+1] - ts_list[i]).total_seconds() for i in range(len(ts_list)-1)]
            diffs = [d for d in diffs if 0 < d < 86400]  # 0-24 saat arası
            avg_time = np.mean(diffs) if diffs else 0
        else:
            avg_time = 0

        # Saat entropisi
        hour_counts = ip_hours.get(ip, {})
        h_entropy = distribution_entropy(hour_counts)
        unique_hours = len(hour_counts)

        # Oturum süresi
        durations = ip_durations.get(ip, [])
        dur_mean = np.mean(durations) if durations else 0

        records.append({
            "src_ip": ip,
            "total_attempts": total,
            "unique_usernames": unique_u,
            "unique_passwords": unique_p,
            "unique_combos": min(unique_u * unique_p, total),  # Üst sınır
            "success_rate": success / max(total, 1),
            "avg_time_between": avg_time,
            "session_count": ip_sessions.get(ip, 0),
            "commands_per_session": ip_commands.get(ip, 0) / max(ip_sessions.get(ip, 0), 1),
            "unique_hours": unique_hours,
            "hour_entropy": h_entropy,
            "avg_password_entropy": ip_pass_entropy.get(ip, 0),
            "avg_password_length": ip_pass_length.get(ip, 0),
            "password_reuse_ratio": total / max(unique_p, 1),
            "duration_mean": dur_mean,
            "has_file_download": 1 if ip_downloads.get(ip, 0) > 0 else 0,
        })

    # Bellek temizliği
    del ip_total_login, ip_success, ip_failed, ip_sessions, ip_commands
    del ip_downloads, ip_hours, ip_unique_users, ip_unique_passes
    del ip_timestamps, ip_durations, ip_pass_entropy, ip_pass_length
    gc.collect()

    df = pd.DataFrame(records)
    print(f"    Cowrie: {len(df):,} IP × {len(df.columns)-1} özellik")
    return df


# =====================================================================
# 1b. ÖZELLİK MÜHENDİSLİĞİ — CTU-HORNET
# =====================================================================

def extract_ctu_features(ctu_path):
    """CTU-Hornet parquet'ten IP başına 5 ağ akışı özelliği çıkarır."""
    print("\n  CTU-Hornet özellik mühendisliği...")
    cols = _get_columns(ctu_path)

    ip_flows       = Counter()
    ip_dst_ports   = defaultdict(set)
    ip_dst_ips     = defaultdict(set)
    ip_durations   = defaultdict(list)
    ip_protos      = defaultdict(set)

    pf = pq.ParquetFile(ctu_path)
    read_cols = ["src_ip", "dst_port", "dst_ip"]
    if "duration" in cols: read_cols.append("duration")
    if "proto" in cols: read_cols.append("proto")

    batch_num = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        df = batch.to_pandas()

        ip_flows.update(df["src_ip"].dropna().value_counts().to_dict())

        for ip, grp in df.groupby("src_ip"):
            ip_dst_ports[ip].update(grp["dst_port"].dropna().unique())
            ip_dst_ips[ip].update(grp["dst_ip"].dropna().unique())
            if "proto" in grp.columns:
                ip_protos[ip].update(grp["proto"].dropna().unique())
            if "duration" in grp.columns:
                durs = grp["duration"].dropna().tolist()[:100]
                ip_durations[ip].extend(durs)

        del df; gc.collect()
        batch_num += 1
        if batch_num % 20 == 0:
            print(f"      batch {batch_num}...")

    # En az 5 akışı olan IP'ler
    active_ips = [ip for ip, cnt in ip_flows.items() if cnt >= 5]
    print(f"    {len(ip_flows):,} IP → {len(active_ips):,} aktif IP (≥5 akış)")

    records = []
    for ip in active_ips:
        durs = ip_durations.get(ip, [])
        records.append({
            "src_ip": ip,
            "flow_count": ip_flows[ip],
            "unique_dst_ports": len(ip_dst_ports.get(ip, set())),
            "unique_dst_ips": len(ip_dst_ips.get(ip, set())),
            "avg_flow_duration": np.mean(durs) if durs else 0,
            "proto_diversity": len(ip_protos.get(ip, set())),
        })

    del ip_flows, ip_dst_ports, ip_dst_ips, ip_durations, ip_protos
    gc.collect()

    df = pd.DataFrame(records)
    print(f"    CTU: {len(df):,} IP × {len(df.columns)-1} özellik")
    return df


# =====================================================================
# 2. ÖN İŞLEME
# =====================================================================

def preprocess_features(df, feature_cols):
    """Log dönüşümü + StandardScaler normalizasyonu."""
    print("\n  Ön işleme...")

    df_feat = df[feature_cols].copy()

    # Log dönüşümü (sağa çarpık sütunlar)
    log_cols = [c for c in feature_cols if df_feat[c].max() > 100]
    for c in log_cols:
        df_feat[c] = np.log1p(df_feat[c])
    print(f"    Log dönüşümü: {len(log_cols)} sütun")

    # NaN/Inf temizliği
    df_feat = df_feat.replace([np.inf, -np.inf], 0).fillna(0)

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat)
    print(f"    StandardScaler: {X_scaled.shape}")

    return X_scaled, scaler, log_cols


# =====================================================================
# 3. K-MEANS KÜMELEME
# =====================================================================

def find_optimal_k(X, k_range=range(2, 11)):
    """Elbow + Silhouette + Davies-Bouldin + Calinski-Harabasz (çoklu metrik konsensüsü)."""
    

    print("\n  Optimal k aranıyor (çoklu metrik)...")
    inertias, sils, dbs, chs = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
        dbs.append(davies_bouldin_score(X, labels))
        chs.append(calinski_harabasz_score(X, labels))
        print(f"    k={k}: inertia={km.inertia_:>12,.0f} | "
              f"sil={sils[-1]:.4f} | DB={dbs[-1]:.4f} | CH={chs[-1]:>10,.0f}")

    # CSV'ye kaydet
    metrics_df = pd.DataFrame({
        'k': list(k_range), 'Inertia': inertias,
        'Silhouette': sils, 'Davies_Bouldin': dbs,
        'Calinski_Harabasz': chs
    })
    metrics_df.to_csv(os.path.join(WEEK5_DATA, "cluster_metrics_comparison.csv"),
                      index=False)
    print(f"    → data/cluster_metrics_comparison.csv")

    # Konsensüs sıralaması: Silhouette↑, DB↓, CH↑
    sil_rank = np.argsort(sils)[::-1]
    db_rank = np.argsort(dbs)
    ch_rank = np.argsort(chs)[::-1]
    consensus = {}
    for i, k in enumerate(k_range):
        idx_in = list(k_range).index(k)
        consensus[k] = (list(sil_rank).index(idx_in) +
                        list(db_rank).index(idx_in) +
                        list(ch_rank).index(idx_in))
    best_k = min(consensus, key=consensus.get)
    print(f"    Konsensüs: en iyi k = {best_k} "
          f"(sıralama toplamı={consensus[best_k]}, düşük=iyi)")

    return list(k_range), inertias, sils, dbs, chs, best_k


def run_kmeans(X, k, random_state=42):
    """K-Means çalıştırma."""
    print(f"\n  K-Means (k={k}) eğitiliyor...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=500)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"    Silhouette: {sil:.4f}")
    print(f"    Küme boyutları: {dict(Counter(labels))}")
    return km, labels, sil


def label_clusters(km, feature_cols, scaler, log_cols):
    """
    Küme merkezlerini analiz ederek Bot/Manuel/Hibrit etiketi atar.
    Bot göstergeleri: yüksek deneme, düşük süre, düşük entropi, yüksek sabit saat
    Manuel göstergeleri: düşük deneme, yüksek süre, yüksek entropi, değişken saatler
    """
    centers = km.cluster_centers_

    # Orijinal ölçeğe geri dönüştür
    centers_original = scaler.inverse_transform(centers)
    for i, c in enumerate(feature_cols):
        if c in log_cols:
            centers_original[:, i] = np.expm1(centers_original[:, i])

    centers_df = pd.DataFrame(centers_original, columns=feature_cols)

    # Bot skoru hesapla (yüksek = daha çok bot)
    bot_scores = []
    for idx, row in centers_df.iterrows():
        score = 0
        # Yüksek deneme → bot
        if "total_attempts" in feature_cols:
            score += np.log1p(row.get("total_attempts", 0)) / 10
        # Düşük denemeler arası süre → bot (hızlı)
        if "avg_time_between" in feature_cols:
            avg_t = row.get("avg_time_between", 1)
            score += 1 / (1 + avg_t / 10)
        # Yüksek şifre tekrar oranı → bot
        if "password_reuse_ratio" in feature_cols:
            score += min(row.get("password_reuse_ratio", 1), 100) / 100
        # Düşük şifre entropisi → bot
        if "avg_password_entropy" in feature_cols:
            score += max(0, 3 - row.get("avg_password_entropy", 3)) / 3
        # Yüksek saat entropisi (tüm saatlerde aktif) → bot
        if "hour_entropy" in feature_cols:
            score += row.get("hour_entropy", 0) / 4
        # Düşük komut → bot (sadece login dener)
        if "commands_per_session" in feature_cols:
            score += 1 / (1 + row.get("commands_per_session", 0))

        bot_scores.append(score)

    # Sıralama ile etiketleme
    sorted_idx = np.argsort(bot_scores)[::-1]
    labels_map = {}

    if len(sorted_idx) == 2:
        labels_map[sorted_idx[0]] = "Bot"
        labels_map[sorted_idx[1]] = "Manuel"
    elif len(sorted_idx) == 3:
        labels_map[sorted_idx[0]] = "Bot"
        labels_map[sorted_idx[1]] = "Hibrit"
        labels_map[sorted_idx[2]] = "Manuel"
    else:
        for i, idx in enumerate(sorted_idx):
            if i < len(sorted_idx) // 3:
                labels_map[idx] = "Bot"
            elif i < 2 * len(sorted_idx) // 3:
                labels_map[idx] = "Hibrit"
            else:
                labels_map[idx] = "Manuel"

    print(f"    Bot skorları: {dict(zip(range(len(bot_scores)), [f'{s:.2f}' for s in bot_scores]))}")
    print(f"    Etiketler: {labels_map}")

    return labels_map, centers_df, bot_scores


# =====================================================================
# 4. GÖRSELLEŞTİRME
# =====================================================================

def fig_01_elbow(k_range, inertias, silhouettes, dbs, chs, best_k, save_dir):
    print("\n  [1/8] Çoklu metrik grafiği...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(k_range, inertias, "o-", color=C_DARK, linewidth=2, markersize=6)
    axes[0, 0].set_xlabel("Küme Sayısı (k)"); axes[0, 0].set_ylabel("Inertia (WCSS)")
    axes[0, 0].set_title("Elbow Yöntemi"); axes[0, 0].set_xticks(k_range)

    axes[0, 1].plot(k_range, silhouettes, "s-", color=C_BOT, linewidth=2, markersize=6)
    axes[0, 1].set_xlabel("Küme Sayısı (k)"); axes[0, 1].set_ylabel("Silhouette")
    axes[0, 1].set_title("Silhouette (yüksek = iyi)"); axes[0, 1].set_xticks(k_range)
    axes[0, 1].axvline(best_k, color="green", linestyle="--", alpha=0.7,
                       label=f"Konsensüs k={best_k}")
    axes[0, 1].legend()

    axes[1, 0].plot(k_range, dbs, "^-", color=C_MANUAL, linewidth=2, markersize=6)
    axes[1, 0].set_xlabel("Küme Sayısı (k)"); axes[1, 0].set_ylabel("Davies-Bouldin")
    axes[1, 0].set_title("Davies-Bouldin (düşük = iyi)"); axes[1, 0].set_xticks(k_range)
    axes[1, 0].axvline(best_k, color="green", linestyle="--", alpha=0.7)

    axes[1, 1].plot(k_range, chs, "d-", color=C_HYBRID, linewidth=2, markersize=6)
    axes[1, 1].set_xlabel("Küme Sayısı (k)"); axes[1, 1].set_ylabel("Calinski-Harabasz")
    axes[1, 1].set_title("Calinski-Harabasz (yüksek = iyi)"); axes[1, 1].set_xticks(k_range)
    axes[1, 1].axvline(best_k, color="green", linestyle="--", alpha=0.7)

    plt.suptitle(f"Optimal Küme Sayısı — Çoklu Metrik Analizi (Konsensüs: k={best_k})",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_cluster_metrics.png")); plt.close(fig)
    print(f"    → 01_cluster_metrics.png")


def fig_02_pca_clusters(X, labels, labels_map, save_dir):
    print("\n  [2/8] PCA 2D küme görselleştirmesi...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    cluster_colors = {}
    color_list = [C_BOT, C_MANUAL, C_HYBRID, "#9B59B6", "#1ABC9C"]
    for cid, name in labels_map.items():
        if "Bot" in name: cluster_colors[cid] = C_BOT
        elif "Manuel" in name: cluster_colors[cid] = C_MANUAL
        else: cluster_colors[cid] = C_HYBRID

    fig, ax = plt.subplots(figsize=(10, 8))

    for cid in sorted(labels_map.keys()):
        mask = labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=cluster_colors.get(cid, "#999"), alpha=0.4, s=8,
                   label=f"{labels_map[cid]} (n={mask.sum():,})", edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Saldırgan Kümeleri — PCA 2D Projeksiyon")
    ax.legend(fontsize=10, markerscale=3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_pca_clusters.png")); plt.close(fig)
    print(f"    → 02_pca_clusters.png")
    return pca


def fig_03_silhouette_detail(X, labels, labels_map, save_dir):
    print("\n  [3/8] Silhouette detay grafiği...")
    sil_vals = silhouette_samples(X, labels)
    avg_sil = np.mean(sil_vals)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    n_clusters = len(labels_map)

    cluster_colors = {}
    for cid, name in labels_map.items():
        if "Bot" in name: cluster_colors[cid] = C_BOT
        elif "Manuel" in name: cluster_colors[cid] = C_MANUAL
        else: cluster_colors[cid] = C_HYBRID

    for cid in sorted(labels_map.keys()):
        cluster_sil = sil_vals[labels == cid]
        cluster_sil.sort()
        size = len(cluster_sil)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=cluster_colors.get(cid, "#999"), alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, labels_map[cid], fontsize=9, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color="red", linestyle="--", label=f"Ortalama: {avg_sil:.3f}")
    ax.set_xlabel("Silhouette Değeri")
    ax.set_ylabel("Saldırgan IP")
    ax.set_title("Küme Bazlı Silhouette Analizi")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_silhouette_detail.png")); plt.close(fig)
    print(f"    → 03_silhouette_detail.png")


def fig_04_cluster_profiles(centers_df, labels_map, feature_cols, save_dir):
    print("\n  [4/8] Küme profilleri radar chart...")

    # Normalize (0-1 arası)
    norm_df = centers_df[feature_cols].copy()
    for c in norm_df.columns:
        rng = norm_df[c].max() - norm_df[c].min()
        if rng > 0:
            norm_df[c] = (norm_df[c] - norm_df[c].min()) / rng
        else:
            norm_df[c] = 0.5

    # Radar chart için en ayırt edici 8 özellik
    variance = norm_df.var()
    top_features = variance.nlargest(min(8, len(feature_cols))).index.tolist()

    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    color_map = {}
    for cid, name in labels_map.items():
        if "Bot" in name: color_map[cid] = C_BOT
        elif "Manuel" in name: color_map[cid] = C_MANUAL
        else: color_map[cid] = C_HYBRID

    for cid in sorted(labels_map.keys()):
        values = norm_df.loc[cid, top_features].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=labels_map[cid],
                color=color_map.get(cid, "#999"), markersize=5)
        ax.fill(angles, values, alpha=0.1, color=color_map.get(cid, "#999"))

    ax.set_xticks(angles[:-1])
    short_labels = [f.replace("_", "\n") for f in top_features]
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_title("Küme Profilleri — Radar Chart", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_cluster_profiles_radar.png")); plt.close(fig)
    print(f"    → 04_cluster_profiles_radar.png")


def fig_05_feature_importance(centers_df, labels_map, feature_cols, save_dir):
    print("\n  [5/8] Özellik önem sıralaması...")

    # Her özellik için küme merkezleri arasındaki varyans
    importance = {}
    for c in feature_cols:
        vals = centers_df[c].values
        rng = vals.max() - vals.min()
        mean = np.mean(np.abs(vals)) + 1e-10
        importance[c] = rng / mean

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    labels_f = [x[0] for x in sorted_imp]
    values_f = [x[1] for x in sorted_imp]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels_f)))
    ax.barh(range(len(labels_f)), values_f, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels_f)))
    ax.set_yticklabels(labels_f, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Ayırt Edicilik Skoru")
    ax.set_title("Özellik Önem Sıralaması (Küme Merkezleri Farkı)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_feature_importance.png")); plt.close(fig)
    print(f"    → 05_feature_importance.png")


def fig_06_cluster_stats(df, labels_map, feature_cols, save_dir):
    print("\n  [6/8] Küme istatistik tablosu...")

    fig, ax = plt.subplots(figsize=(14, max(4, len(feature_cols) * 0.35 + 2)))
    ax.axis("off")

    cluster_names = [labels_map[i] for i in sorted(labels_map.keys())]
    header = ["Özellik"] + [f"{name}\n(n={int((df['cluster_label']==name).sum()):,})" for name in cluster_names]

    table_data = [header]
    for feat in feature_cols:
        row = [feat]
        for name in cluster_names:
            mask = df["cluster_label"] == name
            val = df.loc[mask, feat].mean()
            if val >= 1000:
                row.append(f"{val:,.0f}")
            elif val >= 1:
                row.append(f"{val:.1f}")
            else:
                row.append(f"{val:.3f}")
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[0.25] + [0.75/len(cluster_names)] * len(cluster_names))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    for j in range(len(header)):
        table[0, j].set_facecolor(C_DARK)
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            table[i, j].set_facecolor("#f8f8f8" if i % 2 == 0 else "white")

    ax.set_title("Küme Bazlı Ortalama Özellik Değerleri", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_cluster_statistics.png")); plt.close(fig)
    print(f"    → 06_cluster_statistics.png")


def fig_07_bot_vs_manual(df, save_dir):
    print("\n  [7/8] Bot vs Manuel davranış karşılaştırması...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    comparisons = [
        ("total_attempts", "Toplam Deneme Sayısı", True),
        ("avg_time_between", "Denemeler Arası Süre (sn)", True),
        ("password_reuse_ratio", "Şifre Tekrar Oranı", True),
        ("avg_password_entropy", "Ortalama Şifre Entropisi", False),
        ("unique_hours", "Aktif Saat Çeşitliliği", False),
        ("commands_per_session", "Oturum Başına Komut", True),
    ]

    cluster_colors = {"Bot": C_BOT, "Manuel": C_MANUAL, "Hibrit": C_HYBRID}

    for idx, (feat, title, use_log) in enumerate(comparisons):
        ax = axes[idx // 3][idx % 3]
        if feat not in df.columns:
            ax.text(0.5, 0.5, "Veri yok", transform=ax.transAxes, ha="center")
            ax.set_title(title)
            continue

        for label in df["cluster_label"].unique():
            data = df.loc[df["cluster_label"] == label, feat].dropna()
            if len(data) == 0: continue
            data_plot = np.log1p(data) if use_log else data
            ax.hist(data_plot, bins=50, alpha=0.5, label=label,
                    color=cluster_colors.get(label, "#999"), edgecolor="none")

        ax.set_title(title)
        ax.set_ylabel("IP Sayısı")
        if use_log:
            ax.set_xlabel(f"log₁₊({feat})")
        ax.legend(fontsize=8)

    plt.suptitle("Bot vs Manuel Saldırgan Davranış Karşılaştırması",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "07_bot_vs_manual.png")); plt.close(fig)
    print(f"    → 07_bot_vs_manual.png")


def fig_08_cluster_distribution(df, save_dir):
    print("\n  [8/8] Küme dağılımı pasta grafiği...")

    cluster_counts = df["cluster_label"].value_counts()
    cluster_colors = {"Bot": C_BOT, "Manuel": C_MANUAL, "Hibrit": C_HYBRID}
    colors = [cluster_colors.get(name, "#999") for name in cluster_counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # IP sayısı bazlı
    ax1.pie(cluster_counts.values, labels=cluster_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=90,
            textprops={"fontsize": 11})
    ax1.set_title("IP Sayısına Göre")

    # Toplam deneme bazlı
    attempt_by_cluster = df.groupby("cluster_label")["total_attempts"].sum()
    attempt_by_cluster = attempt_by_cluster.reindex(cluster_counts.index)
    colors2 = [cluster_colors.get(name, "#999") for name in attempt_by_cluster.index]
    ax2.pie(attempt_by_cluster.values, labels=attempt_by_cluster.index,
            autopct="%1.1f%%", colors=colors2, startangle=90,
            textprops={"fontsize": 11})
    ax2.set_title("Toplam Deneme Sayısına Göre")

    plt.suptitle("Küme Dağılımı", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "08_cluster_distribution.png")); plt.close(fig)
    print(f"    → 08_cluster_distribution.png")


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 62 + "╗")
    print("║  HAFTA 5: DAVRANIŞSAL KÜMELEME — Bot vs Manuel (v7)       ║")
    print("╚" + "═" * 62 + "╝")
    t_start = time.time()

    # Ön kontrol
    for p, name in [(COWRIE_PATH, "Cowrie"), (CTU_PATH, "CTU-Hornet")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} bulunamadı: {p}")
        pf = pq.ParquetFile(p)
        print(f"  {name}: {pf.metadata.num_rows:,} satır")

    os.makedirs(WEEK5_FIGS, exist_ok=True)
    os.makedirs(WEEK5_DATA, exist_ok=True)

     # ═══════════════════════════════════════════════════════════
    # 1. ÖZELLİK MÜHENDİSLİĞİ (cache'li)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("1. ÖZELLİK MÜHENDİSLİĞİ")
    print("=" * 62)

    cache_path = os.path.join(WEEK5_DATA, "attacker_features.csv")
    if os.path.exists(cache_path):
        print(f"\n  ✓ Cache bulundu: {cache_path}")
        features_df = pd.read_csv(cache_path)
        print(f"    Yüklendi: {len(features_df):,} IP × {len(features_df.columns)-1} özellik")
    else:
        print(f"\n  Cache yok, ham veriden çıkarılıyor...")
        cowrie_features = extract_cowrie_features(COWRIE_PATH)
        ctu_features = extract_ctu_features(CTU_PATH)
        features_df = cowrie_features.copy()
        common_ips = set(features_df["src_ip"]) & set(ctu_features["src_ip"])
        if len(common_ips) > 100:
            print(f"\n  {len(common_ips):,} ortak IP — CTU özellikleri ekleniyor")
            features_df = features_df.merge(ctu_features, on="src_ip", how="left",
                                            suffixes=("", "_ctu"))
        else:
            print(f"\n  Ortak IP az ({len(common_ips)}) — sadece Cowrie özellikleri kullanılacak")
        features_df.to_csv(cache_path, index=False)
        print(f"    → data/attacker_features.csv kaydedildi")

    feature_cols = [c for c in features_df.columns
                    if c not in ("src_ip", "cluster_id", "cluster_label")
                    and features_df[c].dtype in ("float64", "int64", "int32", "float32")]
    print(f"\n  Kümeleme özellikleri ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"    {c}: min={features_df[c].min():.2f}, "
              f"max={features_df[c].max():.2f}, mean={features_df[c].mean():.2f}")
    # ═══════════════════════════════════════════════════════════
    # 2. ÖN İŞLEME
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("2. ÖN İŞLEME")
    print("=" * 62)

    X_scaled, scaler, log_cols = preprocess_features(features_df, feature_cols)

    # ═══════════════════════════════════════════════════════════
    # 3. K-MEANS KÜMELEME
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("3. K-MEANS KÜMELEME")
    print("=" * 62)

    k_range, inertias, silhouettes, dbs, chs, best_k = find_optimal_k(X_scaled)
    fig_01_elbow(k_range, inertias, silhouettes, dbs, chs, best_k, WEEK5_FIGS)

    # Kümeleme çalıştır
    km, labels, sil_score = run_kmeans(X_scaled, best_k)

    # Küme etiketleme
    labels_map, centers_df, bot_scores = label_clusters(km, feature_cols, scaler, log_cols)

    # Etiketleri DataFrame'e ekle
    features_df["cluster_id"] = labels
    features_df["cluster_label"] = features_df["cluster_id"].map(labels_map)

    # ═══════════════════════════════════════════════════════════
    # 4. GÖRSELLEŞTİRME
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("4. GÖRSELLEŞTİRME")
    print("=" * 62)

    pca = fig_02_pca_clusters(X_scaled, labels, labels_map, WEEK5_FIGS)
    fig_03_silhouette_detail(X_scaled, labels, labels_map, WEEK5_FIGS)
    fig_04_cluster_profiles(centers_df, labels_map, feature_cols, WEEK5_FIGS)
    fig_05_feature_importance(centers_df, labels_map, feature_cols, WEEK5_FIGS)
    fig_06_cluster_stats(features_df, labels_map, feature_cols, WEEK5_FIGS)
    fig_07_bot_vs_manual(features_df, WEEK5_FIGS)
    fig_08_cluster_distribution(features_df, WEEK5_FIGS)

    # ═══════════════════════════════════════════════════════════
    # SONUÇLARI KAYDET
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("SONUÇLAR")
    print("=" * 62)

    # Etiketli veri
    features_df.to_csv(os.path.join(WEEK5_DATA, "clustered_attackers.csv"), index=False)
    print(f"  → data/clustered_attackers.csv")

    # Küme özeti
    summary = features_df.groupby("cluster_label")[feature_cols].agg(["mean", "median", "std", "count"])
    summary.to_csv(os.path.join(WEEK5_DATA, "cluster_summary.csv"))
    print(f"  → data/cluster_summary.csv")

    # PCA bileşenleri
    pca_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    pca_df.to_csv(os.path.join(WEEK5_DATA, "pca_variance.csv"), index=False)

    # Konsol özeti
    print(f"\n  Küme Dağılımı:")
    for label in sorted(features_df["cluster_label"].unique()):
        mask = features_df["cluster_label"] == label
        n = mask.sum()
        total_att = features_df.loc[mask, "total_attempts"].sum()
        print(f"    {label}: {n:,} IP ({n/len(features_df)*100:.1f}%), {total_att:,} toplam deneme")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 62}")
    print(f"HAFTA 5 TAMAMLANDI — {elapsed:.0f}s")
    print(f"\n  Figürler ({WEEK5_FIGS}/):")
    for f in sorted(os.listdir(WEEK5_FIGS)):
        print(f"    {f}")
    print(f"\n  Veri ({WEEK5_DATA}/):")
    for f in sorted(os.listdir(WEEK5_DATA)):
        print(f"    {f}")
    print(f"{'=' * 62}")