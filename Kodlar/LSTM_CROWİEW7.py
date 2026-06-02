"""
W7 Cowrie — LSTM Sekans (SEQ_LEN=50)
========================================================================
Sadece Hafta 7 Cowrie erken uyarı görevini SEQ_LEN=50 ile test eder.
Amaç: SEQ_LEN=20'de F1=0.67 olan sonucu iyileştirmek.
"""

import os
import gc
import gzip
import json
import math
import time
import warnings
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ijson

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, precision_score,
                             recall_score)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR = "/mnt/c/TEZ_PROJEEE"
COWRIE_RAW_DIR = os.path.join(BASE_DIR, "data", "Cowrie")
LABELED_W7 = os.path.join(BASE_DIR, "output", "week7", "data",
                          "cowrie_early_features.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "lstm_seq_w7cowrie_seq50")
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

SEQ_LEN = 50  # ← BURASI DEĞİŞTİ
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 256
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("╔══════════════════════════════════════════════════════════╗")
print(f"║  W7 Cowrie — LSTM Sekans (SEQ_LEN={SEQ_LEN})                 ║")
print("╚══════════════════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════
# YARDIMCI
# ══════════════════════════════════════════════════════════════
COWRIE_EVENT_MAP = {
    'cowrie.login.success': 1, 'cowrie.login.failed': 2,
    'cowrie.session.connect': 3, 'cowrie.session.closed': 4,
    'cowrie.command.input': 5, 'cowrie.command.failed': 6,
    'cowrie.client.version': 7,
}


def shannon_entropy(s):
    if not s: return 0.0
    cnt = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())


def extract_cowrie_features(ev, prev_ts):
    ts = ev.get('timestamp', '')
    try:
        t = pd.Timestamp(ts).timestamp()
    except Exception:
        t = 0.0
    delta = max(0.0, t - prev_ts) if prev_ts else 0.0
    delta_log = math.log1p(delta)
    try:
        hour = pd.Timestamp(ts).hour if ts else 0
    except Exception:
        hour = 0
    user = str(ev.get('username', '') or '')
    pwd = str(ev.get('password', '') or '')
    eid = ev.get('eventid', '')
    ev_code = COWRIE_EVENT_MAP.get(eid, 0)
    success = 1 if eid == 'cowrie.login.success' else 0
    return [
        delta_log, hour / 23.0, len(user) / 32.0, len(pwd) / 32.0,
        shannon_entropy(pwd) / 8.0, success, ev_code / 7.0,
    ], t


def iter_cowrie_events(file_paths):
    """ijson streaming — RAM dostu."""
    for fp in file_paths:
        try:
            with gzip.open(fp, 'rb') as gz:
                for session_dict in ijson.items(gz, 'item'):
                    for sid, events in session_dict.items():
                        for ev in events:
                            ip = ev.get('src_ip_identifier')
                            if ip:
                                yield ip, ev, fp
            gc.collect()
        except Exception as e:
            print(f"      ⚠ Dosya hatası ({fp}): {e}")
            continue


def build_sequences(event_iter, valid_ips, seq_len, total_files):
    sequences = {}
    prev_ts = {}
    filled = set()
    n_total = 0
    n_kept = 0
    last_fp = None
    file_count = 0
    t_start = time.time()

    for ip, ev, fp in event_iter:
        if fp != last_fp:
            file_count += 1
            last_fp = fp
            if file_count % 10 == 0 or file_count == total_files:
                el = time.time() - t_start
                print(f"      {file_count}/{total_files} dosya | "
                      f"okunan:{n_total:,} | tutulan:{n_kept:,} | "
                      f"dolu IP: {len(filled)}/{len(valid_ips)} | {el:.1f}s")
        n_total += 1
        if ip not in valid_ips or ip in filled:
            continue
        feats, t = extract_cowrie_features(ev, prev_ts.get(ip, 0.0))
        prev_ts[ip] = t
        sequences.setdefault(ip, []).append(feats)
        n_kept += 1
        if len(sequences[ip]) >= seq_len:
            filled.add(ip)
            if len(filled) == len(valid_ips):
                print(f"      Tüm IP'ler doldu, erken bitiş.")
                break

    n_features = len(next(iter(sequences.values()))[0]) if sequences else 0
    out = {}
    for ip in valid_ips:
        seq = sequences.get(ip, [])
        if not seq:
            continue
        seq = seq[:seq_len]
        if len(seq) < seq_len:
            pad = [[0.0] * n_features] * (seq_len - len(seq))
            seq = pad + seq
        out[ip] = seq
    return out, n_features


# ══════════════════════════════════════════════════════════════
# ETİKET YÜKLEME
# ══════════════════════════════════════════════════════════════
print("\n[1/4] Etiket yükleme")
df_lbl = pd.read_csv(LABELED_W7)
ip_col = 'src_ip'
lbl_col = 'threat_label' if 'threat_label' in df_lbl.columns else \
          ('threat_level' if 'threat_level' in df_lbl.columns else 'label')
labels_map = dict(zip(df_lbl[ip_col].astype(str), df_lbl[lbl_col].astype(str)))
valid_ips = set(labels_map.keys())
print(f"   W7 Cowrie etiketli IP: {len(valid_ips):,}")
print(f"   Sınıf dağılımı:")
for k, v in Counter(labels_map.values()).items():
    print(f"     {k}: {v:,}")


# ══════════════════════════════════════════════════════════════
# SEKANS İNŞASI
# ══════════════════════════════════════════════════════════════
print(f"\n[2/4] Sekans inşası (SEQ_LEN={SEQ_LEN})")
cowrie_files = sorted(glob(os.path.join(COWRIE_RAW_DIR, "cyberlab_*.json.gz")))
print(f"   {len(cowrie_files)} dosya bulundu")

seq_dict, n_features = build_sequences(
    iter_cowrie_events(cowrie_files), valid_ips, SEQ_LEN, len(cowrie_files))
print(f"   Sekans oluşturulan IP: {len(seq_dict):,} | feature/event: {n_features}")


# ══════════════════════════════════════════════════════════════
# TENSÖR + NORMALIZE + SPLIT
# ══════════════════════════════════════════════════════════════
print(f"\n[3/4] Tensör hazırlığı")
X, y = [], []
for ip, seq in seq_dict.items():
    if ip in labels_map:
        X.append(seq)
        y.append(labels_map[ip])
X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"   X şekli: {X.shape}")
print(f"   Sınıflar: {dict(Counter(y))}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def normalize_3d(X_train, X_test):
    n_train, T, F = X_train.shape
    flat = X_train.reshape(-1, F)
    mask = (flat != 0).any(axis=1)
    mu = flat[mask].mean(axis=0)
    sd = flat[mask].std(axis=0) + 1e-8
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd
    pad_mask_tr = (X_train.sum(axis=2) == 0)
    pad_mask_te = (X_test.sum(axis=2) == 0)
    Xtr[pad_mask_tr] = 0.0
    Xte[pad_mask_te] = 0.0
    return Xtr, Xte


X_tr, X_te = normalize_3d(X_tr, X_te)
print(f"   Train: {len(X_tr):,} | Test: {len(X_te):,}")


# ══════════════════════════════════════════════════════════════
# MODEL EĞİTİM
# ══════════════════════════════════════════════════════════════
print(f"\n[4/4] LSTM eğitim")

le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr)
y_te_enc = le.transform(y_te)
n_classes = len(le.classes_)
class_names = sorted(le.classes_.tolist())

y_tr_cat = to_categorical(y_tr_enc, n_classes)
y_te_cat = to_categorical(y_te_enc, n_classes)

cw_raw = compute_class_weight('balanced',
                              classes=np.unique(y_tr_enc), y=y_tr_enc)
cw_smooth = np.sqrt(cw_raw)
cw_smooth = cw_smooth / cw_smooth.mean()
class_weight_dict = dict(zip(np.unique(y_tr_enc), cw_smooth))
print(f"   Class weights: "
      f"{ {int(k): round(float(v),3) for k,v in class_weight_dict.items()} }")


def build_model(model_type="standard"):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(SEQ_LEN, n_features)))
    if model_type == "bidirectional":
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(32)))
    else:
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    lr = 5e-4 if model_type == "bidirectional" else 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


results = {}
for model_type in ["standard", "bidirectional"]:
    name = "LSTM" if model_type == "standard" else "Bi-LSTM"
    print(f"\n   [{name}] Eğitiliyor...")
    model = build_model(model_type)

    es = EarlyStopping(monitor='val_loss', patience=12, min_delta=1e-4,
                       restore_best_weights=True, verbose=0)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=5, min_lr=1e-6, verbose=0)

    t = time.time()
    hist = model.fit(X_tr, y_tr_cat, validation_split=0.2,
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     class_weight=class_weight_dict,
                     callbacks=[es, rl], verbose=0)
    train_t = time.time() - t

    t = time.time()
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    pred_t = time.time() - t

    acc = accuracy_score(y_te_enc, y_pred)
    f1 = f1_score(y_te_enc, y_pred, average='weighted')
    prec = precision_score(y_te_enc, y_pred, average='weighted')
    rec = recall_score(y_te_enc, y_pred, average='weighted')

    print(f"     Acc: {acc:.4f} | F1: {f1:.4f} | "
          f"Eğitim: {train_t:.1f}s | Epoch: {len(hist.history['loss'])}")

    results[name] = dict(accuracy=acc, f1=f1, precision=prec, recall=rec,
                         train_time=train_t, predict_time=pred_t,
                         epochs=len(hist.history['loss']))

    # Report
    rep = classification_report(y_te_enc, y_pred, target_names=class_names,
                                output_dict=True)
    pd.DataFrame(rep).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "data",
                     f"w7_cowrie_seq50_{model_type}_report.csv"))

    # Confusion
    cm = confusion_matrix(y_te_enc, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f"Confusion — {name}\nW7 Cowrie (SEQ_LEN={SEQ_LEN})")
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures",
                             f"w7_cowrie_seq50_{model_type}_confusion.png"),
                dpi=150)
    plt.close()

    # History
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(hist.history['loss'], label='train')
    a1.plot(hist.history['val_loss'], label='val')
    a1.set_title(f"{name} Loss"); a1.legend()
    a2.plot(hist.history['accuracy'], label='train')
    a2.plot(hist.history['val_accuracy'], label='val')
    a2.set_title(f"{name} Acc"); a2.legend()
    plt.suptitle(f"W7 Cowrie SEQ_LEN={SEQ_LEN} — {name}", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures",
                             f"w7_cowrie_seq50_{model_type}_history.png"),
                dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# ÖZET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"ÖZET — W7 Cowrie SEQ_LEN={SEQ_LEN}")
print("=" * 60)
print(f"\n  {'Model':<10} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'Eğit(s)':>10}")
print("  " + "─" * 56)
for m, r in results.items():
    print(f"  {m:<10} {r['f1']:>8.4f} {r['accuracy']:>8.4f} "
          f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['train_time']:>10.1f}")

# CSV
rows = []
for m, r in results.items():
    rows.append(dict(Model=m, SEQ_LEN=SEQ_LEN,
                     F1=round(r['f1'], 4), Accuracy=round(r['accuracy'], 4),
                     Precision=round(r['precision'], 4),
                     Recall=round(r['recall'], 4),
                     Train_Time_s=round(r['train_time'], 1),
                     Epochs=r['epochs']))
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "data",
                                       "w7_cowrie_seq50_summary.csv"),
                          index=False)

# Karşılaştırma (SEQ_LEN=20 vs SEQ_LEN=50)
print("\n  Karşılaştırma (SEQ_LEN=20 → 50):")
print(f"    LSTM       SEQ_LEN=20: F1=0.6698  →  SEQ_LEN=50: F1={results['LSTM']['f1']:.4f}  "
      f"(Δ={results['LSTM']['f1']-0.6698:+.4f})")
print(f"    Bi-LSTM    SEQ_LEN=20: F1=0.6409  →  SEQ_LEN=50: F1={results['Bi-LSTM']['f1']:.4f}  "
      f"(Δ={results['Bi-LSTM']['f1']-0.6409:+.4f})")

print(f"\n  Çıktılar: {OUTPUT_DIR}")
print("  ✓ Tamamlandı.")