"""
Hafta 7: Erken Uyarı Sistemi — Saldırgan Davranış Tahmini (v7 uyumlu)
=======================================================================
Bir saldırganın İLK birkaç hareketine bakarak, sonraki davranışını tahmin eder.
Bu, gerçek zamanlı erken uyarı mekanizmasının simülasyonudur.

Temel Soru:
  "Bir IP honeypot'a bağlanıp ilk 3-5 login denemesini yaptığında,
   bu saldırgan sonra komut çalıştıracak mı, dosya indirecek mi,
   yoksa sadece brute-force yapıp çıkacak mı?"

Pipeline:
  ═══════════════════════════════════════════════════════════
  BÖLÜM A — COWRIE ERKEN TESPİT
  ═══════════════════════════════════════════════════════════
    1. Her IP'nin kronolojik oturum geçmişini çıkar
    2. Her IP için:
       - ERKEN ÖZELLİKLER (ilk %30 aktivite):
         İlk N login denemesinden çıkarılan metrikler
       - ETİKET (nihai davranış — tüm oturum boyunca):
         Tehdit seviyesi: Düşük / Orta / Yüksek
         • Düşük:  Sadece bağlan-çık (session.connect + closed, login yok/az)
         • Orta:   Brute-force (çok login denemesi ama komut/dosya yok)
         • Yüksek: Exploitation (login.success + command.input veya file_download)
    3. Model eğitimi ve değerlendirme
    4. Erken tespit penceresi analizi (N=3,5,10,20,50)

  ═══════════════════════════════════════════════════════════
  BÖLÜM B — CTU-HORNET ERKEN TESPİT
  ═══════════════════════════════════════════════════════════
    1. Her IP'nin kronolojik akış geçmişini çıkar
    2. İlk 10 akıştan özellik → nihai davranış etiketi
       • Keşif:    Çok port, az veri
       • Hedefli:  Az port, belirli servislere yoğunlaşma
       • Agresif:  Yüksek hacim, hızlı tarama
    3. Model eğitimi ve değerlendirme

  ═══════════════════════════════════════════════════════════
  BÖLÜM C — ERKEN TESPİT PENCERESİ ANALİZİ
  ═══════════════════════════════════════════════════════════
    Kaç deneme sonra güvenilir tahmin yapılabilir?
    N=3,5,10,20,50 için F1 skorlarını karşılaştır

  Tüm çıktılar → OUTPUT_DIR/week7/
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

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  UYARI: xgboost kurulu değil — pip install xgboost")

warnings.filterwarnings("ignore")

# =====================================================================
# YAPILANDIRMA
# =====================================================================

OUTPUT_DIR  = "/mnt/c/TEZ_PROJEEE/output"
WEEK7_DIR   = os.path.join(OUTPUT_DIR, "week7")
WEEK7_FIGS  = os.path.join(WEEK7_DIR, "figures")
WEEK7_DATA  = os.path.join(WEEK7_DIR, "data")

COWRIE_PATH = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")
CTU_PATH    = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")

BATCH_SIZE  = 500_000
RS = 42

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 10, "axes.titlesize": 14,
    "axes.labelsize": 12, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
})

C_LOW = "#27AE60"; C_MED = "#F18F01"; C_HIGH = "#E74C3C"
C_DARK = "#2C3E50"; C_BLUE = "#2E86AB"


# =====================================================================
# YARDIMCI
# =====================================================================

def _get_cols(p):
    s = pq.read_schema(p)
    return [s.field(i).name for i in range(len(s))]

def _entropy(s):
    if not s or len(s) == 0: return 0.0
    f = Counter(s); n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in f.values())

def _fmt(x, _=None):
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# =====================================================================
# COWRIE — KRONOLOJİK OTURUM GEÇMİŞİ ÇIKARIMI
# =====================================================================

def extract_cowrie_temporal(cowrie_path, early_ratio=0.3):
    """
    Her IP için kronolojik event geçmişini çıkarır.
    İlk %30 aktiviteden ERKEN ÖZELLİKLER,
    tüm aktiviteden NİHAİ DAVRANIŞ ETİKETİ üretir.
    """
    print(f"\n  Cowrie kronolojik özellik çıkarımı (erken oran: {early_ratio})...")
    cols = _get_cols(cowrie_path)

    # ── IP bazlı kronolojik event toplama ──
    # Her IP için: login listesi (timestamp sıralı) + session bilgileri
    ip_login_ts = defaultdict(list)       # ip → [(timestamp, username, password)]
    ip_total_logins = Counter()
    ip_success_count = Counter()
    ip_command_count = Counter()
    ip_download_count = Counter()
    ip_session_count = Counter()
    ip_durations = defaultdict(list)

    pf = pq.ParquetFile(cowrie_path)
    rc = ["eventid", "src_ip", "hour", "timestamp"]
    for c in ["username", "password", "duration"]:
        if c in cols: rc.append(c)

    bn = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=rc):
        df = batch.to_pandas()

        # Session sayısı
        sc = df["eventid"].astype(str).str.contains("session.connect", na=False)
        if sc.any():
            ip_session_count.update(df.loc[sc, "src_ip"].dropna().value_counts().to_dict())

        # Komut sayısı
        cc = df["eventid"].astype(str).str.contains("command.input", na=False)
        if cc.any():
            ip_command_count.update(df.loc[cc, "src_ip"].dropna().value_counts().to_dict())

        # Dosya indirme
        dc = df["eventid"].astype(str).str.contains("file_download", na=False)
        if dc.any():
            ip_download_count.update(df.loc[dc, "src_ip"].dropna().value_counts().to_dict())

        # Duration
        if "duration" in df.columns:
            cl = df["eventid"].astype(str).str.contains("session.closed", na=False)
            for ip, g in df.loc[cl, ["src_ip", "duration"]].dropna().groupby("src_ip"):
                ip_durations[ip].extend(g["duration"].tolist()[:50])

        # Login eventleri — kronolojik bilgi
        lm = df["eventid"].astype(str).str.contains("login", na=False)
        logins = df[lm].copy()
        if len(logins) > 0:
            # Başarılı login sayısı
            sm = logins["eventid"].astype(str).str.contains("success", na=False)
            ip_success_count.update(logins.loc[sm, "src_ip"].dropna().value_counts().to_dict())
            ip_total_logins.update(logins["src_ip"].dropna().value_counts().to_dict())

            # Kronolojik login bilgisi (ilk N için)
            for ip, g in logins.groupby("src_ip"):
                if len(ip_login_ts[ip]) < 200:  # Bellek koruması
                    for _, row in g.head(200 - len(ip_login_ts[ip])).iterrows():
                        ip_login_ts[ip].append({
                            "ts": row.get("timestamp"),
                            "user": row.get("username", ""),
                            "pwd": row.get("password", ""),
                            "success": "success" in str(row.get("eventid", "")),
                        })

        del df, logins; gc.collect()
        bn += 1
        if bn % 30 == 0: print(f"      batch {bn}...")

    # ── Her IP için erken özellik + nihai etiket oluştur ──
    print("    Erken özellikler ve etiketler oluşturuluyor...")

    # Minimum 10 login denemesi olan IP'ler (erken tespit için yeterli veri)
    active_ips = [ip for ip in ip_total_logins if ip_total_logins[ip] >= 10]
    print(f"    {len(ip_total_logins):,} IP → {len(active_ips):,} aktif (≥10 login)")

    records = []
    for ip in active_ips:
        login_list = ip_login_ts.get(ip, [])
        total = ip_total_logins.get(ip, 0)

        # Login listesini timestamp'e göre sırala
        login_list.sort(key=lambda x: x["ts"] if x["ts"] is not None else pd.Timestamp.min)

        # Erken pencere: ilk %30
        n_early = max(3, int(len(login_list) * early_ratio))
        early = login_list[:n_early]
        all_logins = login_list

        # ── ERKEN ÖZELLİKLER (ilk %30 login'den) ──
        early_users = set(e.get("user", "") for e in early if e.get("user"))
        early_pwds = set(e.get("pwd", "") for e in early if e.get("pwd"))
        early_success = sum(1 for e in early if e.get("success", False))

        # Denemeler arası süre (erken dönem)
        early_ts = [e["ts"] for e in early if e["ts"] is not None]
        early_ts.sort()
        if len(early_ts) >= 2:
            diffs = []
            for i in range(len(early_ts) - 1):
                d = (early_ts[i+1] - early_ts[i]).total_seconds()
                if 0 < d < 86400:
                    diffs.append(d)
            early_avg_time = np.mean(diffs) if diffs else 0
            early_min_time = min(diffs) if diffs else 0
            early_max_time = max(diffs) if diffs else 0
            early_std_time = np.std(diffs) if len(diffs) > 1 else 0
        else:
            early_avg_time = early_min_time = early_max_time = early_std_time = 0

        # Şifre entropi (erken dönem)
        if early_pwds:
            early_pwd_entropy = np.mean([_entropy(str(p)) for p in early_pwds])
            early_pwd_len = np.mean([len(str(p)) for p in early_pwds])
        else:
            early_pwd_entropy = early_pwd_len = 0

        # Saat çeşitliliği (erken dönem)
        early_hours = set()
        for e in early:
            if e["ts"] is not None:
                try:
                    early_hours.add(e["ts"].hour)
                except Exception:
                    pass

        # Erken başarı oranı
        early_success_rate = early_success / max(len(early), 1)

        # Tekrar oranı (erken)
        early_reuse = len(early) / max(len(early_pwds), 1)

        # ── NİHAİ DAVRANIŞ ETİKETİ (tüm oturum geçmişinden) ──
        total_commands = ip_command_count.get(ip, 0)
        total_downloads = ip_download_count.get(ip, 0)
        total_success = ip_success_count.get(ip, 0)
        total_sessions = ip_session_count.get(ip, 0)
        durs = ip_durations.get(ip, [])
        avg_dur = np.mean(durs) if durs else 0

        # Tehdit seviyesi belirleme
        has_exploitation = (total_commands > 0 or total_downloads > 0) and total_success > 0
        has_bruteforce = total > 20 and total_success == 0
        has_successful_access = total_success > 0 and total_commands == 0

        if has_exploitation:
            threat_label = "Yüksek"   # Komut çalıştırma veya dosya indirme
        elif has_successful_access or (total > 50 and avg_dur > 5):
            threat_label = "Orta"     # Başarılı giriş ama exploitation yok
        else:
            threat_label = "Düşük"    # Sadece brute-force veya bağlan-çık

        records.append({
            "src_ip": ip,
            # ── ERKEN ÖZELLİKLER (model bunları görecek) ──
            "early_attempt_count": len(early),
            "early_unique_users": len(early_users),
            "early_unique_passwords": len(early_pwds),
            "early_success_rate": early_success_rate,
            "early_avg_time_between": early_avg_time,
            "early_min_time_between": early_min_time,
            "early_std_time_between": early_std_time,
            "early_password_entropy": early_pwd_entropy,
            "early_password_length": early_pwd_len,
            "early_password_reuse": early_reuse,
            "early_unique_hours": len(early_hours),
            # ── NİHAİ DAVRANIŞ ETİKETİ (model bunu TAHMİN edecek) ──
            "threat_label": threat_label,
            # ── Referans (modele verilmez) ──
            "_total_logins": total,
            "_total_commands": total_commands,
            "_total_downloads": total_downloads,
            "_total_success": total_success,
            "_avg_duration": avg_dur,
        })

    del ip_login_ts, ip_total_logins, ip_success_count
    del ip_command_count, ip_download_count, ip_session_count, ip_durations
    gc.collect()

    df = pd.DataFrame(records)
    print(f"    Cowrie: {len(df):,} IP × {len([c for c in df.columns if c.startswith('early_')])} erken özellik")
    print(f"    Tehdit dağılımı:")
    for lbl, cnt in df["threat_label"].value_counts().items():
        print(f"      {lbl}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    return df


# =====================================================================
# CTU-HORNET — KRONOLOJİK AKIŞ GEÇMİŞİ
# =====================================================================

def extract_ctu_temporal(ctu_path, early_n=10):
    """
    Her IP'nin ilk N akışından erken özellik,
    tüm akışlardan nihai davranış etiketi çıkarır.
    """
    print(f"\n  CTU-Hornet kronolojik özellik çıkarımı (ilk {early_n} akış)...")
    cols = _get_cols(ctu_path)

    # IP bazlı akış toplama
    ip_early_ports = defaultdict(list)      # ip → ilk N akışın dst_port listesi
    ip_early_durations = defaultdict(list)  # ip → ilk N akışın duration listesi
    ip_early_protos = defaultdict(list)     # ip → ilk N akışın proto listesi
    ip_early_count = Counter()

    ip_total_flows = Counter()
    ip_all_ports = defaultdict(set)
    ip_all_ips = defaultdict(set)
    ip_all_durations = defaultdict(list)
    ip_all_protos = defaultdict(set)

    pf = pq.ParquetFile(ctu_path)
    rc = ["src_ip", "dst_port", "dst_ip"]
    if "duration" in cols: rc.append("duration")
    if "proto" in cols: rc.append("proto")
    if "timestamp" in cols: rc.append("timestamp")

    bn = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=rc):
        df = batch.to_pandas()

        for ip, g in df.groupby("src_ip"):
            current_count = ip_early_count[ip]
            ip_total_flows[ip] += len(g)

            # Tüm akış bilgileri (etiket için)
            ip_all_ports[ip].update(g["dst_port"].dropna().unique())
            ip_all_ips[ip].update(g["dst_ip"].dropna().unique())
            if "proto" in g.columns:
                ip_all_protos[ip].update(g["proto"].dropna().unique())
            if "duration" in g.columns:
                ip_all_durations[ip].extend(g["duration"].dropna().tolist()[:50])

            # Erken akışlar (ilk N)
            if current_count < early_n:
                remaining = early_n - current_count
                early_g = g.head(remaining)
                ip_early_ports[ip].extend(early_g["dst_port"].dropna().tolist())
                if "duration" in early_g.columns:
                    ip_early_durations[ip].extend(early_g["duration"].dropna().tolist())
                if "proto" in early_g.columns:
                    ip_early_protos[ip].extend(early_g["proto"].dropna().tolist())
                ip_early_count[ip] += len(early_g)

        del df; gc.collect()
        bn += 1
        if bn % 20 == 0: print(f"      batch {bn}...")

    # En az 20 akışı olan IP'ler
    active = [ip for ip, c in ip_total_flows.items() if c >= 20]
    print(f"    {len(ip_total_flows):,} IP → {len(active):,} aktif (≥20 akış)")

    records = []
    for ip in active:
        e_ports = ip_early_ports.get(ip, [])
        e_durs = ip_early_durations.get(ip, [])
        e_protos = ip_early_protos.get(ip, [])

        # ── ERKEN ÖZELLİKLER (ilk N akıştan) ──
        early_unique_ports = len(set(e_ports))
        early_avg_dur = np.mean(e_durs) if e_durs else 0
        early_std_dur = np.std(e_durs) if len(e_durs) > 1 else 0
        early_proto_div = len(set(e_protos))
        early_port_entropy = _entropy([str(p) for p in e_ports]) if e_ports else 0

        # Hedeflenen port kategorileri (erken dönem)
        common_ports = {22, 23, 80, 443, 445, 3389, 8080, 8443}
        early_common_ratio = sum(1 for p in e_ports if p in common_ports) / max(len(e_ports), 1)

        # ── NİHAİ DAVRANIŞ ETİKETİ (tüm akışlardan) ──
        total = ip_total_flows[ip]
        all_unique_ports = len(ip_all_ports.get(ip, set()))
        all_unique_ips = len(ip_all_ips.get(ip, set()))
        all_durs = ip_all_durations.get(ip, [])
        all_avg_dur = np.mean(all_durs) if all_durs else 0

        # Davranış sınıflandırması
        port_scan = all_unique_ports > 20           # Çok port tarama
        high_volume = total > 500                    # Yüksek hacim
        targeted = all_unique_ports <= 5 and total > 10  # Az porta yoğunlaşma

        if port_scan and high_volume:
            behavior = "Agresif"      # Yoğun port tarama
        elif port_scan:
            behavior = "Keşif"        # Port tarama (keşif aşaması)
        elif targeted:
            behavior = "Hedefli"      # Belirli servislere odaklanma
        else:
            behavior = "Keşif"        # Varsayılan

        records.append({
            "src_ip": ip,
            # ── ERKEN ÖZELLİKLER ──
            "early_unique_ports": early_unique_ports,
            "early_avg_duration": early_avg_dur,
            "early_std_duration": early_std_dur,
            "early_proto_diversity": early_proto_div,
            "early_port_entropy": early_port_entropy,
            "early_common_port_ratio": early_common_ratio,
            # ── ETİKET ──
            "behavior_label": behavior,
            # ── Referans ──
            "_total_flows": total,
            "_all_unique_ports": all_unique_ports,
            "_all_unique_ips": all_unique_ips,
            "_all_avg_duration": all_avg_dur,
        })

    del ip_early_ports, ip_early_durations, ip_early_protos, ip_early_count
    del ip_total_flows, ip_all_ports, ip_all_ips, ip_all_durations, ip_all_protos
    gc.collect()

    df = pd.DataFrame(records)
    print(f"    CTU: {len(df):,} IP × {len([c for c in df.columns if c.startswith('early_')])} erken özellik")
    print(f"    Davranış dağılımı:")
    for lbl, cnt in df["behavior_label"].value_counts().items():
        print(f"      {lbl}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    return df


# =====================================================================
# ERKEN TESPİT PENCERESİ ANALİZİ — COWRIE
# =====================================================================

def cowrie_window_analysis(cowrie_path, windows=[3, 5, 10, 20, 50]):
    """
    Farklı erken tespit pencerelerinde (ilk N deneme) model performansını ölçer.
    Soru: Kaç deneme sonra güvenilir tahmin yapılabilir?
    """
    print(f"\n  Erken tespit penceresi analizi (N={windows})...")
    cols = _get_cols(cowrie_path)

    # Tüm IP'lerin kronolojik login geçmişini topla (bir kez)
    ip_logins = defaultdict(list)
    ip_commands = Counter()
    ip_downloads = Counter()
    ip_success = Counter()

    pf = pq.ParquetFile(cowrie_path)
    rc = ["eventid", "src_ip", "timestamp"]
    for c in ["username", "password"]:
        if c in cols: rc.append(c)

    bn = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=rc):
        df = batch.to_pandas()

        cc = df["eventid"].astype(str).str.contains("command.input", na=False)
        if cc.any(): ip_commands.update(df.loc[cc, "src_ip"].dropna().value_counts().to_dict())

        dc = df["eventid"].astype(str).str.contains("file_download", na=False)
        if dc.any(): ip_downloads.update(df.loc[dc, "src_ip"].dropna().value_counts().to_dict())

        lm = df["eventid"].astype(str).str.contains("login", na=False)
        logins = df[lm]
        if len(logins) > 0:
            sm = logins["eventid"].astype(str).str.contains("success", na=False)
            ip_success.update(logins.loc[sm, "src_ip"].dropna().value_counts().to_dict())

            for ip, g in logins.groupby("src_ip"):
                if len(ip_logins[ip]) < 100:
                    for _, row in g.head(100 - len(ip_logins[ip])).iterrows():
                        ip_logins[ip].append({
                            "ts": row.get("timestamp"),
                            "pwd": str(row.get("password", "")),
                        })

        del df, logins; gc.collect()
        bn += 1
        if bn % 30 == 0: print(f"      batch {bn}...")

    # Etiket oluştur (her IP için sabit — nihai davranış)
    # En az max(windows) login'i olan IP'ler
    min_logins = max(windows) + 5
    active_ips = [ip for ip in ip_logins if len(ip_logins[ip]) >= min_logins]
    print(f"    {len(active_ips):,} IP (≥{min_logins} login)")

    ip_labels = {}
    for ip in active_ips:
        has_exploit = (ip_commands.get(ip, 0) > 0 or ip_downloads.get(ip, 0) > 0) and ip_success.get(ip, 0) > 0
        if has_exploit:
            ip_labels[ip] = "Yüksek"
        elif ip_success.get(ip, 0) > 0:
            ip_labels[ip] = "Orta"
        else:
            ip_labels[ip] = "Düşük"

    # Her pencere boyutu için özellik çıkar + model eğit
    window_results = {}

    for N in windows:
        print(f"\n    Pencere N={N}...")
        records = []
        for ip in active_ips:
            login_list = sorted(ip_logins[ip], key=lambda x: x["ts"] if x["ts"] is not None else pd.Timestamp.min)
            early = login_list[:N]

            # Erken özellikler
            pwds = [e["pwd"] for e in early if e["pwd"]]
            unique_pwds = set(pwds)

            ts_list = [e["ts"] for e in early if e["ts"] is not None]
            ts_list.sort()
            if len(ts_list) >= 2:
                diffs = [(ts_list[i+1]-ts_list[i]).total_seconds() for i in range(len(ts_list)-1)]
                diffs = [d for d in diffs if 0 < d < 86400]
                avg_t = np.mean(diffs) if diffs else 0
                std_t = np.std(diffs) if len(diffs) > 1 else 0
            else:
                avg_t = std_t = 0

            pwd_ent = np.mean([_entropy(p) for p in unique_pwds]) if unique_pwds else 0
            pwd_len = np.mean([len(p) for p in unique_pwds]) if unique_pwds else 0

            records.append({
                "src_ip": ip,
                "unique_passwords": len(unique_pwds),
                "avg_time_between": avg_t,
                "std_time_between": std_t,
                "password_entropy": pwd_ent,
                "password_length": pwd_len,
                "password_reuse": N / max(len(unique_pwds), 1),
                "label": ip_labels[ip],
            })

        wdf = pd.DataFrame(records)
        feat_cols = ["unique_passwords", "avg_time_between", "std_time_between",
                     "password_entropy", "password_length", "password_reuse"]

        X = wdf[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
        for c in feat_cols:
            if X[c].max() > 100: X[c] = np.log1p(X[c])

        le = LabelEncoder()
        y = le.fit_transform(wdf["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RS, stratify=y)

        # Sadece Random Forest (hız için)
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RS, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        f1_w = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)

        window_results[N] = {
            "f1": f1_w, "accuracy": acc,
            "report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
            "le": le,
        }
        print(f"      F1={f1_w:.4f}, Acc={acc:.4f}")

    del ip_logins, ip_commands, ip_downloads, ip_success
    gc.collect()

    return window_results


# =====================================================================
# DENETİMLİ SINIFLANDIRMA (Multi-class)
# =====================================================================

def train_multiclass(X_train, X_test, y_train, y_test, le, ds_name):
    print(f"\n  {ds_name} — Multi-class model eğitimi...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    models = {}; results = {}
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train); Xes = scaler.transform(X_test)

    n_classes = len(le.classes_)

    # RF
    print(f"    [RF] Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RS, n_jobs=-1)
    rf.fit(X_train, y_train)
    rfp = rf.predict(X_test)
    rfc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_weighted")
    models["Random Forest"] = rf
    results["Random Forest"] = {"pred": rfp, "cv": rfc}
    print(f"      CV F1: {rfc.mean():.4f}±{rfc.std():.4f} | Test: {accuracy_score(y_test, rfp):.4f}")

    # XGBoost
    if HAS_XGB:
        print(f"    [XGB] XGBoost...")
        xg = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=RS, eval_metric="mlogloss", use_label_encoder=False, n_jobs=-1)
        xg.fit(X_train, y_train)
        xgp = xg.predict(X_test)
        xgc = cross_val_score(xg, X_train, y_train, cv=cv, scoring="f1_weighted")
        models["XGBoost"] = xg
        results["XGBoost"] = {"pred": xgp, "cv": xgc}
        print(f"      CV F1: {xgc.mean():.4f}±{xgc.std():.4f} | Test: {accuracy_score(y_test, xgp):.4f}")

    # SVM
    print(f"    [SVM] SVM...")
    sn = min(10_000, len(Xts))
    if sn < len(Xts):
        si = np.random.default_rng(RS).choice(len(Xts), sn, replace=False)
        Xsv, ysv = Xts[si], y_train[si]
    else:
        Xsv, ysv = Xts, y_train
    svm = SVC(C=10, kernel="rbf", gamma="scale", random_state=RS, probability=False)
    svm.fit(Xsv, ysv)
    svp = svm.predict(Xes)
    svc = cross_val_score(svm, Xsv, ysv, cv=cv, scoring="f1_weighted")
    models["SVM"] = svm
    results["SVM"] = {"pred": svp, "cv": svc}
    print(f"      CV F1: {svc.mean():.4f}±{svc.std():.4f} | Test: {accuracy_score(y_test, svp):.4f}")

    # LR
    print(f"    [LR] Logistic Regression...")
    lr = LogisticRegression(C=1, max_iter=1000, random_state=RS, n_jobs=-1)
    lr.fit(Xts, y_train)
    lrp = lr.predict(Xes)
    lrc = cross_val_score(lr, Xts, y_train, cv=cv, scoring="f1_weighted")
    models["Logistic Regression"] = lr
    results["Logistic Regression"] = {"pred": lrp, "cv": lrc}
    print(f"      CV F1: {lrc.mean():.4f}±{lrc.std():.4f} | Test: {accuracy_score(y_test, lrp):.4f}")

    return models, results


# =====================================================================
# GÖRSELLEŞTİRME
# =====================================================================

def plot_confusion(results, y_test, le, pf, sd):
    n = len(results)
    fig, ax = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: ax = [ax]
    for i, (nm, r) in enumerate(results.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, r["pred"]), display_labels=le.classes_)\
            .plot(ax=ax[i], cmap="Blues", colorbar=False)
        ax[i].set_title(nm, fontsize=11, fontweight="bold")
    plt.suptitle(f"Confusion Matrix — {pf}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_confusion.png")); plt.close(fig)
    print(f"    → {pf}_confusion.png")


def plot_comparison(results, y_test, pf, sd):
    rows = []
    for nm, r in results.items():
        rows.append({"Model": nm,
            "Accuracy": f"{accuracy_score(y_test, r['pred']):.4f}",
            "F1 (weighted)": f"{f1_score(y_test, r['pred'], average='weighted'):.4f}",
            "Precision": f"{precision_score(y_test, r['pred'], average='weighted'):.4f}",
            "Recall": f"{recall_score(y_test, r['pred'], average='weighted'):.4f}",
            "CV F1": f"{r['cv'].mean():.4f}±{r['cv'].std():.4f}"})
    comp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows)*0.8+2))); ax.axis("off")
    td = [list(comp.columns)] + comp.values.tolist()
    t = ax.table(cellText=td, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.8)
    for j in range(len(comp.columns)):
        t[0, j].set_facecolor(C_DARK); t[0, j].set_text_props(color="white", fontweight="bold")
    f1s = [float(r["F1 (weighted)"]) for r in rows]; bi = np.argmax(f1s)+1
    for j in range(len(comp.columns)): t[bi, j].set_facecolor("#D5F5E3")
    for i in range(1, len(td)):
        if i != bi:
            for j in range(len(comp.columns)): t[i, j].set_facecolor("#f8f8f8" if i%2 == 0 else "white")
    ax.set_title(f"Model Karşılaştırması — {pf}", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_comparison.png")); plt.close(fig)
    print(f"    → {pf}_comparison.png")
    return comp


def plot_importance(models, fcols, pf, sd):
    imps = {}
    if "Random Forest" in models: imps["Random Forest"] = models["Random Forest"].feature_importances_
    if "XGBoost" in models: imps["XGBoost"] = models["XGBoost"].feature_importances_
    if not imps: return
    n = len(imps)
    fig, ax = plt.subplots(1, n, figsize=(8*n, 6))
    if n == 1: ax = [ax]
    cm = {"Random Forest": "#E74C3C", "XGBoost": "#F18F01"}
    for i, (nm, imp) in enumerate(imps.items()):
        si = np.argsort(imp)[::-1]
        ax[i].barh(range(len(si)), imp[si], color=cm.get(nm, "#999"), alpha=0.85)
        ax[i].set_yticks(range(len(si))); ax[i].set_yticklabels([fcols[j] for j in si], fontsize=9)
        ax[i].invert_yaxis(); ax[i].set_xlabel("Önem"); ax[i].set_title(nm, fontweight="bold")
    plt.suptitle(f"Özellik Önem Sıralaması — {pf}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_importance.png")); plt.close(fig)
    print(f"    → {pf}_importance.png")


def plot_window_analysis(window_results, sd):
    """Erken tespit penceresi — N vs F1 grafiği."""
    ns = sorted(window_results.keys())
    f1s = [window_results[n]["f1"] for n in ns]
    accs = [window_results[n]["accuracy"] for n in ns]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, f1s, "o-", color=C_HIGH, linewidth=2, markersize=8, label="F1-Score")
    ax.plot(ns, accs, "s--", color=C_BLUE, linewidth=2, markersize=8, label="Accuracy")

    for n, f1, acc in zip(ns, f1s, accs):
        ax.annotate(f"{f1:.3f}", (n, f1), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=C_HIGH, fontweight="bold")

    ax.set_xlabel("Erken Tespit Penceresi (İlk N Login Denemesi)")
    ax.set_ylabel("Skor")
    ax.set_title("Erken Uyarı Sistemi — Kaç Deneme Sonra Güvenilir Tahmin?")
    ax.set_xticks(ns)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    # Referans çizgisi
    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5)
    ax.text(max(ns)*0.95, 0.71, "Kabul edilebilir eşik", ha="right", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(sd, "early_detection_window.png"))
    plt.close(fig)
    print(f"    → early_detection_window.png")


def plot_reports(results, y_test, le, pf, sd):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n))
    if n == 1: axes = [axes]
    for i, (nm, r) in enumerate(results.items()):
        rpt = classification_report(y_test, r["pred"], target_names=le.classes_, output_dict=True)
        rows = []
        for cls in le.classes_:
            c = rpt[cls]
            rows.append([cls, f"{c['precision']:.4f}", f"{c['recall']:.4f}", f"{c['f1-score']:.4f}", f"{int(c['support']):,}"])
        wa = rpt["weighted avg"]
        rows.append(["W.Avg", f"{wa['precision']:.4f}", f"{wa['recall']:.4f}", f"{wa['f1-score']:.4f}", f"{int(wa['support']):,}"])
        axes[i].axis("off")
        td = [["Sınıf", "Precision", "Recall", "F1", "Support"]] + rows
        t = axes[i].table(cellText=td, cellLoc="center", loc="center", colWidths=[0.20, 0.18, 0.18, 0.18, 0.15])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
        for j in range(5):
            t[0, j].set_facecolor(C_DARK); t[0, j].set_text_props(color="white", fontweight="bold")
        axes[i].set_title(nm, fontsize=13, fontweight="bold", pad=15)
    plt.suptitle(f"Classification Report — {pf}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_reports.png")); plt.close(fig)
    print(f"    → {pf}_reports.png")


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 66 + "╗")
    print("║  HAFTA 7: ERKEN UYARI SİSTEMİ — Saldırgan Davranış Tahmini  ║")
    print("╚" + "═" * 66 + "╝")
    t0 = time.time()

    for p, n in [(COWRIE_PATH, "Cowrie"), (CTU_PATH, "CTU-Hornet")]:
        if not os.path.exists(p): raise FileNotFoundError(f"{n}: {p}")
        print(f"  {n}: {pq.ParquetFile(p).metadata.num_rows:,} satır")

    os.makedirs(WEEK7_FIGS, exist_ok=True)
    os.makedirs(WEEK7_DATA, exist_ok=True)

    # ═══ A — COWRIE ERKEN TESPİT ═══
    print("\n" + "=" * 66)
    print("BÖLÜM A — COWRIE ERKEN TESPİT")
    print("  Soru: İlk birkaç login denemesinden tehdit seviyesi tahmin edilebilir mi?")
    print("=" * 66)

    cf = extract_cowrie_temporal(COWRIE_PATH, early_ratio=0.3)

    feat_cols_c = [c for c in cf.columns if c.startswith("early_")]
    Xc = cf[feat_cols_c].replace([np.inf, -np.inf], 0).fillna(0)
    for c in feat_cols_c:
        if Xc[c].max() > 100: Xc[c] = np.log1p(Xc[c])

    lec = LabelEncoder(); yc = lec.fit_transform(cf["threat_label"])
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(Xc, yc, test_size=0.2, random_state=RS, stratify=yc)
    print(f"\n  Train: {len(Xtr_c):,} | Test: {len(Xte_c):,}")

    mc, rc = train_multiclass(Xtr_c, Xte_c, ytr_c, yte_c, lec, "Cowrie Erken Tespit")

    print("\n  Cowrie görselleştirmeleri...")
    plot_confusion(rc, yte_c, lec, "cowrie_early", WEEK7_FIGS)
    comp_c = plot_comparison(rc, yte_c, "cowrie_early", WEEK7_FIGS)
    plot_importance(mc, feat_cols_c, "cowrie_early", WEEK7_FIGS)
    plot_reports(rc, yte_c, lec, "cowrie_early", WEEK7_FIGS)

    # ═══ B — CTU-HORNET ERKEN TESPİT ═══
    print("\n" + "=" * 66)
    print("BÖLÜM B — CTU-HORNET ERKEN TESPİT")
    print("  Soru: İlk 10 akıştan saldırganın davranış tipi tahmin edilebilir mi?")
    print("=" * 66)

    tf = extract_ctu_temporal(CTU_PATH, early_n=10)

    feat_cols_t = [c for c in tf.columns if c.startswith("early_")]
    Xt = tf[feat_cols_t].replace([np.inf, -np.inf], 0).fillna(0)
    for c in feat_cols_t:
        if Xt[c].max() > 100: Xt[c] = np.log1p(Xt[c])

    let = LabelEncoder(); yt = let.fit_transform(tf["behavior_label"])
    Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(Xt, yt, test_size=0.2, random_state=RS, stratify=yt)
    print(f"\n  Train: {len(Xtr_t):,} | Test: {len(Xte_t):,}")

    mt, rt = train_multiclass(Xtr_t, Xte_t, ytr_t, yte_t, let, "CTU-Hornet Erken Tespit")

    print("\n  CTU-Hornet görselleştirmeleri...")
    plot_confusion(rt, yte_t, let, "ctu_early", WEEK7_FIGS)
    comp_t = plot_comparison(rt, yte_t, "ctu_early", WEEK7_FIGS)
    plot_importance(mt, feat_cols_t, "ctu_early", WEEK7_FIGS)
    plot_reports(rt, yte_t, let, "ctu_early", WEEK7_FIGS)

    # ═══ C — ERKEN TESPİT PENCERESİ ANALİZİ ═══
    print("\n" + "=" * 66)
    print("BÖLÜM C — ERKEN TESPİT PENCERESİ ANALİZİ")
    print("  Soru: Kaç login denemesi sonra güvenilir tahmin yapılabilir?")
    print("=" * 66)

    wr = cowrie_window_analysis(COWRIE_PATH, windows=[3, 5, 10, 20, 50])
    plot_window_analysis(wr, WEEK7_FIGS)

    # ═══ SONUÇLAR ═══
    print("\n" + "=" * 66)
    print("SONUÇLAR")
    print("=" * 66)

    comp_c.to_csv(os.path.join(WEEK7_DATA, "cowrie_early_comparison.csv"), index=False)
    comp_t.to_csv(os.path.join(WEEK7_DATA, "ctu_early_comparison.csv"), index=False)
    cf.to_csv(os.path.join(WEEK7_DATA, "cowrie_early_features.csv"), index=False)
    tf.to_csv(os.path.join(WEEK7_DATA, "ctu_early_features.csv"), index=False)

    # Pencere analizi kaydet
    wr_rows = [{"N": n, "F1": r["f1"], "Accuracy": r["accuracy"]} for n, r in sorted(wr.items())]
    pd.DataFrame(wr_rows).to_csv(os.path.join(WEEK7_DATA, "window_analysis.csv"), index=False)

    for ds, mdls, fc in [("cowrie", mc, feat_cols_c), ("ctu", mt, feat_cols_t)]:
        for nm in ["Random Forest", "XGBoost"]:
            if nm in mdls:
                pd.DataFrame({"feature": fc, "importance": mdls[nm].feature_importances_})\
                  .sort_values("importance", ascending=False)\
                  .to_csv(os.path.join(WEEK7_DATA, f"{ds}_imp_{nm.lower().replace(' ', '_')}.csv"), index=False)

    print(f"\n  Cowrie Erken Tespit:\n{comp_c.to_string(index=False)}")
    print(f"\n  CTU-Hornet Erken Tespit:\n{comp_t.to_string(index=False)}")
    print(f"\n  Pencere Analizi (N → F1):")
    for n in sorted(wr.keys()):
        print(f"    N={n:>2}: F1={wr[n]['f1']:.4f}")

    bc = max(rc, key=lambda n: f1_score(yte_c, rc[n]["pred"], average="weighted"))
    bt = max(rt, key=lambda n: f1_score(yte_t, rt[n]["pred"], average="weighted"))
    print(f"\n  En iyi Cowrie: {bc} (F1={f1_score(yte_c, rc[bc]['pred'], average='weighted'):.4f})")
    print(f"  En iyi CTU: {bt} (F1={f1_score(yte_t, rt[bt]['pred'], average='weighted'):.4f})")

    el = time.time() - t0
    print(f"\n{'='*66}\nHAFTA 7 TAMAMLANDI — {el:.0f}s")
    print(f"\n  Figürler ({WEEK7_FIGS}/):")
    for f in sorted(os.listdir(WEEK7_FIGS)):
        if f.endswith(".png"): print(f"    {f}")
    print(f"\n  Veri ({WEEK7_DATA}/):")
    for f in sorted(os.listdir(WEEK7_DATA)):
        print(f"    {f}")
    print(f"{'='*66}")