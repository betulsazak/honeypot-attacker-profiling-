"""
Hafta 6: Denetimli Saldırgan Sınıflandırması — Bağımsız Özellik/Etiket (v3)
==============================================================================
ETİKETLER → Session davranışından türetilir (komut, oturum süresi, dosya indirme)
ÖZELLİKLER → Login davranışından türetilir (hız, şifre çeşitliliği, saat dağılımı)

Bu tasarım etiket sızıntısını (label leakage) önler:
  - Model, etiketin nasıl üretildiğini bilmiyor
  - Etiketler ve özellikler farklı event tiplerinden geliyor
  - Model gerçekten login paternlerinden session davranışını tahmin etmeyi öğreniyor

Pipeline:
  ═══════════════════════════════════════════════════════════
  BÖLÜM A — COWRIE (Bağımsız Etiket/Özellik)
  ═══════════════════════════════════════════════════════════
    ETİKET KAYNAĞI (session davranışı):
      - commands_per_session ≥ 1     → interaktif (Manuel göstergesi)
      - has_file_download = 1        → hedefli (Manuel göstergesi)
      - duration_mean > 10 saniye    → uzun oturum (Manuel göstergesi)
      - success_rate > 0.5           → başarılı giriş (Manuel göstergesi)
      → 4 kriterden 2+ = Manuel, aksi = Bot

    ÖZELLİKLER (login davranışı — etiket üretiminde KULLANILMAZ):
      F1.  total_attempts        : Toplam login denemesi
      F2.  unique_usernames      : Benzersiz kullanıcı adı
      F3.  unique_passwords      : Benzersiz şifre
      F4.  avg_time_between      : Denemeler arası süre
      F5.  unique_hours          : Aktif saat çeşitliliği
      F6.  hour_entropy          : Saat dağılımı entropisi
      F7.  avg_password_entropy  : Şifre entropisi ortalaması
      F8.  avg_password_length   : Şifre uzunluğu ortalaması
      F9.  password_reuse_ratio  : Şifre tekrar oranı
      F10. session_count         : Oturum sayısı (bağlam bilgisi)

  ═══════════════════════════════════════════════════════════
  BÖLÜM B — CTU-HORNET (Bağımsız Etiket/Özellik)
  ═══════════════════════════════════════════════════════════
    ETİKET KAYNAĞI (bağlantı davranışı):
      - avg_flow_duration < 0.5 sn   → bağlan-kop (Bot göstergesi)
      - proto_diversity = 1          → tek protokol (Bot göstergesi)
      → 2/2 = Bot, aksi = Manuel

    ÖZELLİKLER (tarama davranışı — etiket üretiminde KULLANILMAZ):
      F1. flow_count           : Toplam akış
      F2. unique_dst_ports     : Hedef port çeşitliliği
      F3. unique_dst_ips       : Hedef IP çeşitliliği

  ═══════════════════════════════════════════════════════════
  BÖLÜM C — K-MEANS ÇAPRAZ DOĞRULAMA
  ═══════════════════════════════════════════════════════════

  Tüm çıktılar → OUTPUT_DIR/week6/
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
    ConfusionMatrixDisplay, cohen_kappa_score
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
WEEK5_DATA  = os.path.join(OUTPUT_DIR, "week5", "data")
WEEK6_DIR   = os.path.join(OUTPUT_DIR, "week6")
WEEK6_FIGS  = os.path.join(WEEK6_DIR, "figures")
WEEK6_DATA  = os.path.join(WEEK6_DIR, "data")

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

C_BOT = "#E74C3C"; C_MANUAL = "#2E86AB"; C_DARK = "#2C3E50"; C_ACCENT = "#F18F01"


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

def _dist_entropy(d):
    t = sum(d.values())
    if t == 0: return 0.0
    return -sum((c/t)*math.log2(c/t) for c in d.values() if c > 0)

def _fmt(x, _=None):
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# =====================================================================
# COWRIE — ÖZELLİK ve ETİKET ÇIKARIMI (AYRI KAYNAKLAR)
# =====================================================================

def extract_cowrie_independent(cowrie_path):
    """
    Cowrie'den iki bağımsız bilgi seti çıkarır:
      - Etiketler: session davranışından (command, duration, download, success)
      - Özellikler: login davranışından (hız, şifre, saat)
    """
    print("\n  Cowrie bağımsız özellik/etiket çıkarımı...")
    cols = _get_cols(cowrie_path)

    # ── ETİKET kaynağı: Session davranışı ──
    ip_sessions   = Counter()
    ip_commands   = Counter()
    ip_downloads  = Counter()
    ip_success    = Counter()
    ip_total_login = Counter()
    ip_dur        = defaultdict(list)

    # ── ÖZELLİK kaynağı: Login davranışı ──
    ip_hours      = defaultdict(Counter)
    ip_users      = defaultdict(set)
    ip_passes     = defaultdict(set)
    ip_ts         = defaultdict(list)
    MAX_TS = 200

    pf = pq.ParquetFile(cowrie_path)
    rc = ["eventid", "src_ip", "hour"]
    for c in ["username", "password", "timestamp", "duration"]:
        if c in cols: rc.append(c)

    bn = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=rc):
        df = batch.to_pandas()

        # Session davranışı → ETİKET için
        for pat, ctr in [("session.connect", ip_sessions),
                         ("command.input", ip_commands),
                         ("file_download", ip_downloads),
                         ("login.success", ip_success)]:
            m = df["eventid"].astype(str).str.contains(pat, na=False)
            if m.any():
                ctr.update(df.loc[m, "src_ip"].dropna().value_counts().to_dict())

        # Duration → ETİKET için
        if "duration" in df.columns:
            cm = df["eventid"].astype(str).str.contains("session.closed", na=False)
            for ip, g in df.loc[cm, ["src_ip", "duration"]].dropna().groupby("src_ip"):
                ip_dur[ip].extend(g["duration"].tolist()[:100])

        # Login davranışı → ÖZELLİK için
        lm = df["eventid"].astype(str).str.contains("login", na=False)
        logins = df[lm]
        if len(logins) > 0:
            ip_total_login.update(logins["src_ip"].dropna().value_counts().to_dict())

            # Saat dağılımı
            vh = logins[["src_ip", "hour"]].dropna()
            for ip, g in vh.groupby("src_ip"):
                ip_hours[ip].update(g["hour"].astype(int).value_counts().to_dict())

            # Benzersiz user/pass
            if "username" in logins.columns:
                for ip, g in logins[["src_ip", "username"]].dropna().groupby("src_ip"):
                    ip_users[ip].update(g["username"].unique())
            if "password" in logins.columns:
                for ip, g in logins[["src_ip", "password"]].dropna().groupby("src_ip"):
                    ip_passes[ip].update(g["password"].unique())

            # Timestamp → denemeler arası süre
            if "timestamp" in df.columns:
                lts = df.loc[lm, ["src_ip", "timestamp"]].dropna()
                for ip, g in lts.groupby("src_ip"):
                    if len(ip_ts[ip]) < MAX_TS:
                        ip_ts[ip].extend(g["timestamp"].tolist()[:MAX_TS - len(ip_ts[ip])])

        del df, logins; gc.collect()
        bn += 1
        if bn % 30 == 0: print(f"      batch {bn}...")

    # Şifre entropi
    ip_pe = {}; ip_pl = {}
    for ip, ps in ip_passes.items():
        if ps:
            ip_pe[ip] = np.mean([_entropy(str(p)) for p in list(ps)[:500]])
            ip_pl[ip] = np.mean([len(str(p)) for p in list(ps)[:500]])

    # ── DataFrame oluştur: en az 5 login denemesi + en az 1 session ──
    active = [ip for ip in set(ip_total_login) & set(ip_sessions)
              if ip_total_login.get(ip, 0) >= 5 and ip_sessions.get(ip, 0) >= 1]
    print(f"    {len(active):,} aktif IP (≥5 login + ≥1 session)")

    records = []
    for ip in active:
        total = ip_total_login.get(ip, 0)
        u = len(ip_users.get(ip, set()))
        p = len(ip_passes.get(ip, set()))
        success = ip_success.get(ip, 0)
        sessions = ip_sessions.get(ip, 0)
        commands = ip_commands.get(ip, 0)
        downloads = ip_downloads.get(ip, 0)

        # Denemeler arası süre
        ts = sorted(ip_ts.get(ip, []))
        if len(ts) >= 2:
            diffs = [d for d in [(ts[i+1]-ts[i]).total_seconds() for i in range(len(ts)-1)] if 0 < d < 86400]
            avg_t = np.mean(diffs) if diffs else 0
        else:
            avg_t = 0

        hc = ip_hours.get(ip, {})
        durs = ip_dur.get(ip, [])
        dur_mean = np.mean(durs) if durs else 0
        cps = commands / max(sessions, 1)

        # ── ETİKET: Session davranışından (4 kriter) ──
        manuel_score = 0
        manuel_score += int(cps >= 1)                     # Komut çalıştırıyor
        manuel_score += int(downloads > 0)                # Dosya indiriyor
        manuel_score += int(dur_mean > 10)                # Uzun oturum
        manuel_score += int(success / max(total, 1) > 0.5)  # Yüksek başarı oranı
        label = "Manuel" if manuel_score >= 2 else "Bot"

        # ── ÖZELLİKLER: Login davranışından (etiket kriterlerinden BAĞIMSIZ) ──
        records.append({
            "src_ip": ip,
            # ── Özellikler (model bunları görecek) ──
            "total_attempts": total,
            "unique_usernames": u,
            "unique_passwords": p,
            "avg_time_between": avg_t,
            "unique_hours": len(hc),
            "hour_entropy": _dist_entropy(hc),
            "avg_password_entropy": ip_pe.get(ip, 0),
            "avg_password_length": ip_pl.get(ip, 0),
            "password_reuse_ratio": total / max(p, 1),
            "session_count": sessions,
            # ── Etiket (model bunu TAHMİN edecek) ──
            "label": label,
            "manuel_score": manuel_score,
            # ── Referans: etiket kaynağı değerler (kontrol için, modele verilmez) ──
            "_commands_per_session": cps,
            "_has_file_download": int(downloads > 0),
            "_duration_mean": dur_mean,
            "_success_rate": success / max(total, 1),
        })

    del ip_sessions, ip_commands, ip_downloads, ip_success, ip_total_login
    del ip_dur, ip_hours, ip_users, ip_passes, ip_ts, ip_pe, ip_pl
    gc.collect()

    df = pd.DataFrame(records)
    print(f"    Cowrie: {len(df):,} IP")

    # Etiket dağılımı
    for lbl, cnt in df["label"].value_counts().items():
        print(f"      {lbl}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    return df


# =====================================================================
# CTU-HORNET — BAĞIMSIZ ÖZELLİK/ETİKET
# =====================================================================

def extract_ctu_independent(ctu_path):
    """
    CTU-Hornet'ten bağımsız bilgi setleri:
      - Etiket: bağlantı davranışı (duration, proto)
      - Özellik: tarama davranışı (flow count, port/IP çeşitliliği)
    """
    print("\n  CTU-Hornet bağımsız özellik/etiket çıkarımı...")
    cols = _get_cols(ctu_path)

    ip_flows = Counter()
    ip_ports = defaultdict(set); ip_dsts = defaultdict(set)
    ip_dur = defaultdict(list); ip_proto = defaultdict(set)

    pf = pq.ParquetFile(ctu_path)
    rc = ["src_ip", "dst_port", "dst_ip"]
    if "duration" in cols: rc.append("duration")
    if "proto" in cols: rc.append("proto")

    bn = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=rc):
        df = batch.to_pandas()
        ip_flows.update(df["src_ip"].dropna().value_counts().to_dict())
        for ip, g in df.groupby("src_ip"):
            ip_ports[ip].update(g["dst_port"].dropna().unique())
            ip_dsts[ip].update(g["dst_ip"].dropna().unique())
            if "proto" in g.columns: ip_proto[ip].update(g["proto"].dropna().unique())
            if "duration" in g.columns: ip_dur[ip].extend(g["duration"].dropna().tolist()[:100])
        del df; gc.collect()
        bn += 1
        if bn % 20 == 0: print(f"      batch {bn}...")

    active = [ip for ip, c in ip_flows.items() if c >= 5]
    print(f"    {len(active):,} aktif IP")

    records = []
    for ip in active:
        d = ip_dur.get(ip, [])
        avg_dur = np.mean(d) if d else 0
        proto_div = len(ip_proto.get(ip, set()))

        # ── ETİKET: Bağlantı davranışından ──
        bot_score = 0
        bot_score += int(avg_dur < 0.5)       # Çok kısa bağlantı
        bot_score += int(proto_div <= 1)       # Tek protokol
        label = "Bot" if bot_score >= 2 else "Manuel"

        # ── ÖZELLİKLER: Tarama davranışından (etiket kriterlerinden BAĞIMSIZ) ──
        records.append({
            "src_ip": ip,
            "flow_count": ip_flows[ip],
            "unique_dst_ports": len(ip_ports.get(ip, set())),
            "unique_dst_ips": len(ip_dsts.get(ip, set())),
            "label": label,
            "bot_score": bot_score,
            "_avg_flow_duration": avg_dur,
            "_proto_diversity": proto_div,
        })

    del ip_flows, ip_ports, ip_dsts, ip_dur, ip_proto; gc.collect()
    df = pd.DataFrame(records)
    print(f"    CTU: {len(df):,} IP")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"      {lbl}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    return df


# =====================================================================
# DENETİMLİ SINIFLANDIRMA
# =====================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, le, ds_name):
    print(f"\n  {ds_name} — Model eğitimi...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    models = {}; results = {}
    scaler = StandardScaler()
    Xts = scaler.fit_transform(X_train); Xes = scaler.transform(X_test)

    # RF
    print(f"    [RF] Random Forest...")
    rf = GridSearchCV(RandomForestClassifier(random_state=RS, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
        cv=cv, scoring="f1_weighted", n_jobs=-1)
    rf.fit(X_train, y_train); rfm = rf.best_estimator_
    rfp = rfm.predict(X_test); rfpr = rfm.predict_proba(X_test)[:, 1]
    rfc = cross_val_score(rfm, X_train, y_train, cv=cv, scoring="f1_weighted")
    models["Random Forest"] = rfm
    results["Random Forest"] = {"pred": rfp, "proba": rfpr, "cv": rfc, "params": rf.best_params_}
    print(f"      CV F1: {rfc.mean():.4f}±{rfc.std():.4f} | Test Acc: {accuracy_score(y_test, rfp):.4f}")

    # XGBoost
    if HAS_XGB:
        print(f"    [XGB] XGBoost...")
        xg = GridSearchCV(XGBClassifier(random_state=RS, eval_metric="logloss", use_label_encoder=False, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1]},
            cv=cv, scoring="f1_weighted", n_jobs=-1)
        xg.fit(X_train, y_train); xgm = xg.best_estimator_
        xgp = xgm.predict(X_test); xgpr = xgm.predict_proba(X_test)[:, 1]
        xgc = cross_val_score(xgm, X_train, y_train, cv=cv, scoring="f1_weighted")
        models["XGBoost"] = xgm
        results["XGBoost"] = {"pred": xgp, "proba": xgpr, "cv": xgc, "params": xg.best_params_}
        print(f"      CV F1: {xgc.mean():.4f}±{xgc.std():.4f} | Test Acc: {accuracy_score(y_test, xgp):.4f}")

    # SVM
    print(f"    [SVM] Support Vector Machine...")
    sn = min(10_000, len(Xts))
    if sn < len(Xts):
        si = np.random.default_rng(RS).choice(len(Xts), sn, replace=False)
        Xsv, ysv = Xts[si], y_train[si]
    else:
        Xsv, ysv = Xts, y_train
    sv = GridSearchCV(SVC(random_state=RS, probability=True),
        {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
        cv=cv, scoring="f1_weighted", n_jobs=-1)
    sv.fit(Xsv, ysv); svm = sv.best_estimator_
    svp = svm.predict(Xes); svpr = svm.predict_proba(Xes)[:, 1]
    svc = cross_val_score(svm, Xsv, ysv, cv=cv, scoring="f1_weighted")
    models["SVM"] = svm
    results["SVM"] = {"pred": svp, "proba": svpr, "cv": svc, "params": sv.best_params_}
    print(f"      CV F1: {svc.mean():.4f}±{svc.std():.4f} | Test Acc: {accuracy_score(y_test, svp):.4f}")

    # LR
    print(f"    [LR] Logistic Regression...")
    lr = GridSearchCV(LogisticRegression(random_state=RS, max_iter=1000, n_jobs=-1),
        {"C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="f1_weighted", n_jobs=-1)
    lr.fit(Xts, y_train); lrm = lr.best_estimator_
    lrp = lrm.predict(Xes); lrpr = lrm.predict_proba(Xes)[:, 1]
    lrc = cross_val_score(lrm, Xts, y_train, cv=cv, scoring="f1_weighted")
    models["Logistic Regression"] = lrm
    results["Logistic Regression"] = {"pred": lrp, "proba": lrpr, "cv": lrc, "params": lr.best_params_}
    print(f"      CV F1: {lrc.mean():.4f}±{lrc.std():.4f} | Test Acc: {accuracy_score(y_test, lrp):.4f}")

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


def plot_roc(results, y_test, pf, sd):
    fig, ax = plt.subplots(figsize=(9, 7))
    cl = {"Random Forest": "#E74C3C", "XGBoost": "#F18F01", "SVM": "#2E86AB", "Logistic Regression": "#27AE60"}
    for nm, r in results.items():
        fp, tp, _ = roc_curve(y_test, r["proba"])
        ax.plot(fp, tp, label=f"{nm} (AUC={auc(fp, tp):.4f})", color=cl.get(nm, "#999"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Eğrileri — {pf}"); ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_roc.png")); plt.close(fig)
    print(f"    → {pf}_roc.png")


def plot_comparison(results, y_test, pf, sd):
    rows = []
    for nm, r in results.items():
        fp, tp, _ = roc_curve(y_test, r["proba"])
        rows.append({"Model": nm,
            "Accuracy": f"{accuracy_score(y_test, r['pred']):.4f}",
            "Precision": f"{precision_score(y_test, r['pred'], average='weighted'):.4f}",
            "Recall": f"{recall_score(y_test, r['pred'], average='weighted'):.4f}",
            "F1-Score": f"{f1_score(y_test, r['pred'], average='weighted'):.4f}",
            "AUC": f"{auc(fp, tp):.4f}",
            "CV F1": f"{r['cv'].mean():.4f}±{r['cv'].std():.4f}"})
    comp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(15, max(3, len(rows)*0.8+2))); ax.axis("off")
    td = [list(comp.columns)] + comp.values.tolist()
    t = ax.table(cellText=td, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.8)
    for j in range(len(comp.columns)):
        t[0, j].set_facecolor(C_DARK); t[0, j].set_text_props(color="white", fontweight="bold")
    f1s = [float(r["F1-Score"]) for r in rows]; bi = np.argmax(f1s)+1
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
    fig, ax = plt.subplots(1, n, figsize=(8*n, 7))
    if n == 1: ax = [ax]
    cm = {"Random Forest": "#E74C3C", "XGBoost": "#F18F01"}
    for i, (nm, imp) in enumerate(imps.items()):
        si = np.argsort(imp)[::-1]
        ax[i].barh(range(len(si)), imp[si], color=cm.get(nm, "#999"), alpha=0.85)
        ax[i].set_yticks(range(len(si))); ax[i].set_yticklabels([fcols[j] for j in si], fontsize=9)
        ax[i].invert_yaxis(); ax[i].set_xlabel("Önem"); ax[i].set_title(nm, fontweight="bold")
        for bar, val in zip(ax[i].patches, imp[si]):
            ax[i].text(bar.get_width()+max(imp)*0.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    plt.suptitle(f"Özellik Önem Sıralaması — {pf}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_importance.png")); plt.close(fig)
    print(f"    → {pf}_importance.png")


def plot_cv_box(results, pf, sd):
    fig, ax = plt.subplots(figsize=(10, 6))
    cl = {"Random Forest": "#E74C3C", "XGBoost": "#F18F01", "SVM": "#2E86AB", "Logistic Regression": "#27AE60"}
    data = [r["cv"] for r in results.values()]; names = list(results.keys())
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.5)
    for p, nm in zip(bp["boxes"], names): p.set_facecolor(cl.get(nm, "#999")); p.set_alpha(0.7)
    ax.set_ylabel("F1-Score"); ax.set_title(f"5-Fold CV — {pf}")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_cv_boxplot.png")); plt.close(fig)
    print(f"    → {pf}_cv_boxplot.png")


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
        rows.append(["Weighted Avg", f"{wa['precision']:.4f}", f"{wa['recall']:.4f}", f"{wa['f1-score']:.4f}", f"{int(wa['support']):,}"])
        axes[i].axis("off")
        td = [["Sınıf", "Precision", "Recall", "F1-Score", "Support"]] + rows
        t = axes[i].table(cellText=td, cellLoc="center", loc="center", colWidths=[0.20, 0.18, 0.18, 0.18, 0.15])
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
        for j in range(5):
            t[0, j].set_facecolor(C_DARK); t[0, j].set_text_props(color="white", fontweight="bold")
        for j in range(5): t[len(rows), j].set_facecolor("#D5F5E3")
        axes[i].set_title(nm, fontsize=13, fontweight="bold", pad=15)
    plt.suptitle(f"Classification Report — {pf}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_reports.png")); plt.close(fig)
    print(f"    → {pf}_reports.png")


def plot_label_source_analysis(df, pf, sd):
    """Etiket kaynağı ile özellikler arasındaki ilişkiyi gösterir."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pairs = [
        ("total_attempts", "Toplam Deneme"),
        ("avg_time_between", "Denemeler Arası Süre"),
        ("avg_password_entropy", "Şifre Entropisi"),
        ("hour_entropy", "Saat Entropisi"),
    ]

    for idx, (feat, title) in enumerate(pairs):
        ax = axes[idx//2][idx%2]
        if feat not in df.columns: continue
        for lbl, color in [("Bot", C_BOT), ("Manuel", C_MANUAL)]:
            data = df.loc[df["label"] == lbl, feat].dropna()
            if len(data) == 0: continue
            plot_data = np.log1p(data) if data.max() > 100 else data
            ax.hist(plot_data, bins=50, alpha=0.5, label=f"{lbl} (n={len(data):,})",
                    color=color, edgecolor="none")
        ax.set_title(title); ax.legend(fontsize=9); ax.set_ylabel("IP Sayısı")

    plt.suptitle(f"Özellik Dağılımları (Etiket Bazlı) — {pf}\n"
                 f"Not: Etiketler session davranışından, özellikler login davranışından gelir",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout(); plt.savefig(os.path.join(sd, f"{pf}_label_feature_dist.png")); plt.close(fig)
    print(f"    → {pf}_label_feature_dist.png")


# =====================================================================
# ÇAPRAZ DOĞRULAMA — K-Means vs Kural
# =====================================================================

def cross_validate_kmeans(cowrie_df, sd):
    print("\n  Çapraz Doğrulama: Kural vs K-Means...")
    for ext in [".csv", ".parquet"]:
        kp = os.path.join(WEEK5_DATA, f"clustered_attackers{ext}")
        if os.path.exists(kp):
            km_df = pd.read_csv(kp) if ext == ".csv" else pd.read_parquet(kp)
            break
    else:
        print("    Hafta 5 verisi bulunamadı — atlanıyor."); return None

    merged = cowrie_df[["src_ip", "label"]].merge(km_df[["src_ip", "cluster_label"]], on="src_ip", how="inner")
    print(f"    Ortak IP: {len(merged):,}")
    if len(merged) < 100: print("    Yetersiz — atlanıyor."); return None

    agree = (merged["label"] == merged["cluster_label"]).sum()
    kappa = cohen_kappa_score(merged["label"], merged["cluster_label"])
    print(f"    Örtüşme: {agree:,}/{len(merged):,} ({agree/len(merged)*100:.1f}%)")
    print(f"    Cohen's Kappa: {kappa:.4f}")

    labels = ["Bot", "Manuel"]
    cm = confusion_matrix(merged["label"], merged["cluster_label"], labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=axes[0], cmap="Oranges", colorbar=False)
    axes[0].set_xlabel("K-Means Etiketi"); axes[0].set_ylabel("Session-Tabanlı Etiket")
    axes[0].set_title(f"Örtüşme: {agree/len(merged)*100:.1f}% | κ={kappa:.3f}")

    x = np.arange(2); w = 0.35
    rc = merged["label"].value_counts().reindex(labels, fill_value=0)
    kc = merged["cluster_label"].value_counts().reindex(labels, fill_value=0)
    axes[1].bar(x-w/2, rc.values, w, label="Session-Tabanlı", color=C_BOT, alpha=0.85)
    axes[1].bar(x+w/2, kc.values, w, label="K-Means", color=C_MANUAL, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("IP Sayısı"); axes[1].set_title("Dağılım Karşılaştırması"); axes[1].legend()
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))

    plt.suptitle("Çapraz Doğrulama: Session-Tabanlı Etiket vs K-Means",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(sd, "cross_validation.png")); plt.close(fig)
    print(f"    → cross_validation.png")
    return {"agreement": agree/len(merged)*100, "kappa": kappa, "n": len(merged)}


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 66 + "╗")
    print("║  HAFTA 6: DENETİMLİ SINIFLANDIRMA — Bağımsız Etiket v3      ║")
    print("╚" + "═" * 66 + "╝")
    t0 = time.time()

    for p, n in [(COWRIE_PATH, "Cowrie"), (CTU_PATH, "CTU-Hornet")]:
        if not os.path.exists(p): raise FileNotFoundError(f"{n}: {p}")
        print(f"  {n}: {pq.ParquetFile(p).metadata.num_rows:,} satır")

    os.makedirs(WEEK6_FIGS, exist_ok=True)
    os.makedirs(WEEK6_DATA, exist_ok=True)

    # ═══ A — COWRIE ═══
    print("\n" + "=" * 66)
    print("BÖLÜM A — COWRIE")
    print(f"  Etiket kaynağı: Session davranışı (komut, süre, indirme, başarı)")
    print(f"  Özellik kaynağı: Login davranışı (hız, şifre, saat)")
    print("=" * 66)

    cf = extract_cowrie_independent(COWRIE_PATH)

    # Özellik sütunları (etiket kaynağı sütunları _ ile başlıyor, hariç tutuluyor)
    feat_cols_c = [c for c in cf.columns
                   if not c.startswith("_") and c not in ("src_ip", "label", "manuel_score")
                   and cf[c].dtype in ("float64", "int64", "int32", "float32")]

    print(f"\n  Özellikler ({len(feat_cols_c)}): {feat_cols_c}")
    print(f"  Etiket kaynağı sütunları (modele VERİLMEZ): {[c for c in cf.columns if c.startswith('_')]}")

    Xc = cf[feat_cols_c].replace([np.inf, -np.inf], 0).fillna(0)
    for c in feat_cols_c:
        if Xc[c].max() > 100: Xc[c] = np.log1p(Xc[c])

    lec = LabelEncoder(); yc = lec.fit_transform(cf["label"])
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(Xc, yc, test_size=0.2, random_state=RS, stratify=yc)
    print(f"\n  Train: {len(Xtr_c):,} | Test: {len(Xte_c):,}")

    mc, rc = train_and_evaluate(Xtr_c, Xte_c, ytr_c, yte_c, lec, "Cowrie")

    print("\n  Cowrie görselleştirmeleri...")
    plot_confusion(rc, yte_c, lec, "cowrie", WEEK6_FIGS)
    plot_roc(rc, yte_c, "cowrie", WEEK6_FIGS)
    comp_c = plot_comparison(rc, yte_c, "cowrie", WEEK6_FIGS)
    plot_importance(mc, feat_cols_c, "cowrie", WEEK6_FIGS)
    plot_cv_box(rc, "cowrie", WEEK6_FIGS)
    plot_reports(rc, yte_c, lec, "cowrie", WEEK6_FIGS)
    plot_label_source_analysis(cf, "cowrie", WEEK6_FIGS)

    # ═══ B — CTU-HORNET ═══
    print("\n" + "=" * 66)
    print("BÖLÜM B — CTU-HORNET")
    print(f"  Etiket kaynağı: Bağlantı davranışı (süre, protokol)")
    print(f"  Özellik kaynağı: Tarama davranışı (akış, port, IP)")
    print("=" * 66)

    tf = extract_ctu_independent(CTU_PATH)

    feat_cols_t = [c for c in tf.columns
                   if not c.startswith("_") and c not in ("src_ip", "label", "bot_score")
                   and tf[c].dtype in ("float64", "int64", "int32", "float32")]

    print(f"\n  Özellikler ({len(feat_cols_t)}): {feat_cols_t}")

    Xt = tf[feat_cols_t].replace([np.inf, -np.inf], 0).fillna(0)
    for c in feat_cols_t:
        if Xt[c].max() > 100: Xt[c] = np.log1p(Xt[c])

    let = LabelEncoder(); yt = let.fit_transform(tf["label"])
    Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(Xt, yt, test_size=0.2, random_state=RS, stratify=yt)
    print(f"\n  Train: {len(Xtr_t):,} | Test: {len(Xte_t):,}")

    mt, rt = train_and_evaluate(Xtr_t, Xte_t, ytr_t, yte_t, let, "CTU-Hornet")

    print("\n  CTU-Hornet görselleştirmeleri...")
    plot_confusion(rt, yte_t, let, "ctu", WEEK6_FIGS)
    plot_roc(rt, yte_t, "ctu", WEEK6_FIGS)
    comp_t = plot_comparison(rt, yte_t, "ctu", WEEK6_FIGS)
    plot_importance(mt, feat_cols_t, "ctu", WEEK6_FIGS)
    plot_cv_box(rt, "ctu", WEEK6_FIGS)
    plot_reports(rt, yte_t, let, "ctu", WEEK6_FIGS)

    # ═══ C — ÇAPRAZ DOĞRULAMA ═══
    print("\n" + "=" * 66)
    print("BÖLÜM C — ÇAPRAZ DOĞRULAMA")
    print("=" * 66)
    cvr = cross_validate_kmeans(cf, WEEK6_FIGS)

    # ═══ SONUÇLAR ═══
    print("\n" + "=" * 66)
    print("SONUÇLAR")
    print("=" * 66)

    comp_c.to_csv(os.path.join(WEEK6_DATA, "cowrie_model_comparison.csv"), index=False)
    comp_t.to_csv(os.path.join(WEEK6_DATA, "ctu_model_comparison.csv"), index=False)
    cf.to_csv(os.path.join(WEEK6_DATA, "cowrie_labeled.csv"), index=False)
    tf.to_csv(os.path.join(WEEK6_DATA, "ctu_labeled.csv"), index=False)

    for ds, mdls, fc in [("cowrie", mc, feat_cols_c), ("ctu", mt, feat_cols_t)]:
        for nm in ["Random Forest", "XGBoost"]:
            if nm in mdls:
                pd.DataFrame({"feature": fc, "importance": mdls[nm].feature_importances_})\
                  .sort_values("importance", ascending=False)\
                  .to_csv(os.path.join(WEEK6_DATA, f"{ds}_imp_{nm.lower().replace(' ', '_')}.csv"), index=False)

    print(f"\n  Cowrie Sonuçları:\n{comp_c.to_string(index=False)}")
    print(f"\n  CTU-Hornet Sonuçları:\n{comp_t.to_string(index=False)}")
    if cvr: print(f"\n  Çapraz Doğrulama: Örtüşme={cvr['agreement']:.1f}%, κ={cvr['kappa']:.4f}")

    bc = max(rc, key=lambda n: f1_score(yte_c, rc[n]["pred"], average="weighted"))
    bt = max(rt, key=lambda n: f1_score(yte_t, rt[n]["pred"], average="weighted"))
    print(f"\n  En iyi Cowrie: {bc} (F1={f1_score(yte_c, rc[bc]['pred'], average='weighted'):.4f})")
    print(f"  En iyi CTU: {bt} (F1={f1_score(yte_t, rt[bt]['pred'], average='weighted'):.4f})")

    el = time.time() - t0
    print(f"\n{'='*66}\nHAFTA 6 TAMAMLANDI — {el:.0f}s")
    print(f"\n  Figürler ({WEEK6_FIGS}/):")
    for f in sorted(os.listdir(WEEK6_FIGS)):
        if f.endswith(".png"): print(f"    {f}")
    print(f"\n  Veri ({WEEK6_DATA}/):")
    for f in sorted(os.listdir(WEEK6_DATA)):
        print(f"    {f}")
    print(f"{'='*66}")