"""
LSTM SEKANS Modeli — Hafta 6 + Hafta 7
=========================================================================
Per-IP gerçek zaman sekansları üzerinde LSTM/Bi-LSTM.
Ham .gz dosyalarından streaming, RAM dostu (sadece etiketli IP'ler tutulur).

Kullanım:
    python3 week_lstm_sequence.py

PILOT ayarları (CTU): Geo-1, ilk 3 gün. Tüm veri için CONFIG bloğundaki
CTU_HONEYPOTS ve CTU_MAX_DAYS değerlerini None yap.
"""

import os
import gzip
import gc
import ijson
import json
import time
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)
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
DATA_DIR = os.path.join(BASE_DIR, "data")
COWRIE_RAW_DIR = os.path.join(DATA_DIR, "Cowrie")
CTU_ROOT = os.path.join(DATA_DIR, "CTU-Hornet-65-Niner", "zeek")

LABELED_W6_COWRIE = os.path.join(BASE_DIR, "output", "week6", "data", "cowrie_labeled.csv")
LABELED_W6_CTU = os.path.join(BASE_DIR, "output", "week6", "data", "ctu_labeled.csv")
LABELED_W7_COWRIE = os.path.join(BASE_DIR, "output", "week7", "data", "cowrie_early_features.csv")
LABELED_W7_CTU = os.path.join(BASE_DIR, "output", "week7", "data", "ctu_early_features.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "lstm_seq")
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

SEQ_LEN = 20

# --- PILOT: Geo-1, ilk 3 gün ---
CTU_HONEYPOTS = None   # None → tümü
CTU_MAX_DAYS = None     # None → tümü

RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 256
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("╔══════════════════════════════════════════════════════════╗")
print("║  LSTM SEKANS MODELİ — Per-IP Gerçek Zaman Sekansları   ║")
print("╚══════════════════════════════════════════════════════════╝")
print(f"  SEQ_LEN = {SEQ_LEN}")
print(f"  CTU honeypots: {CTU_HONEYPOTS or 'TÜMÜ'}")
print(f"  CTU max gün: {CTU_MAX_DAYS or 'TÜMÜ'}")


# ══════════════════════════════════════════════════════════════
# YARDIMCI: Etiket setleri
# ══════════════════════════════════════════════════════════════
def find_ip_col(df):
    for c in ['src_ip', 'ip', 'source_ip', 'orig_h']:
        if c in df.columns:
            return c
    raise ValueError(f"IP sütunu bulunamadı: {df.columns.tolist()}")


def find_label_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Etiket sütunu bulunamadı. Mevcut: {df.columns.tolist()}")


def load_labels(csv_path, label_candidates):
    df = pd.read_csv(csv_path)
    ip_col = find_ip_col(df)
    lbl_col = find_label_col(df, label_candidates)
    mapping = dict(zip(df[ip_col].astype(str), df[lbl_col].astype(str)))
    return mapping  # {ip: label}


# ══════════════════════════════════════════════════════════════
# YARDIMCI: Feature extractors
# ══════════════════════════════════════════════════════════════
def shannon_entropy(s):
    if not s:
        return 0.0
    cnt = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())


COWRIE_EVENT_MAP = {
    'cowrie.login.success': 1,
    'cowrie.login.failed': 2,
    'cowrie.session.connect': 3,
    'cowrie.session.closed': 4,
    'cowrie.command.input': 5,
    'cowrie.command.failed': 6,
    'cowrie.client.version': 7,
}


def extract_cowrie_features(ev, prev_ts):
    """Tek Cowrie olayından özellik vektörü."""
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

    feats = [
        delta_log,
        hour / 23.0,
        len(user) / 32.0,
        len(pwd) / 32.0,
        shannon_entropy(pwd) / 8.0,
        success,
        ev_code / 7.0,
    ]
    return feats, t


PROTO_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2}
CONN_STATE_MAP = {
    'S0': 0, 'S1': 1, 'SF': 2, 'REJ': 3, 'S2': 4, 'S3': 5,
    'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTRH': 9, 'SH': 10,
    'SHR': 11, 'OTH': 12,
}


def extract_ctu_features(ev, prev_ts):
    """Tek CTU flow olayından özellik vektörü."""
    t = float(ev.get('ts', 0.0))
    delta = max(0.0, t - prev_ts) if prev_ts else 0.0
    delta_log = math.log1p(delta)
    hour = (int(t) // 3600) % 24

    duration = float(ev.get('duration', 0.0) or 0.0)
    orig_b = float(ev.get('orig_bytes', 0.0) or 0.0)
    resp_b = float(ev.get('resp_bytes', 0.0) or 0.0)
    dst_p = float(ev.get('id.resp_p', 0) or 0)
    proto = PROTO_MAP.get(str(ev.get('proto', '')).lower(), 3)
    cstate = CONN_STATE_MAP.get(str(ev.get('conn_state', 'OTH')), 12)

    feats = [
        delta_log,
        hour / 23.0,
        math.log1p(duration),
        math.log1p(orig_b),
        math.log1p(resp_b),
        dst_p / 65535.0,
        proto / 3.0,
        cstate / 12.0,
    ]
    return feats, t


# ══════════════════════════════════════════════════════════════
# YARDIMCI: Event iterators (kaynak formatına özel)
# ══════════════════════════════════════════════════════════════
def iter_cowrie_events(file_paths):
    """Cowrie nested JSON yapısı: [{session_id: [events]}, ...]
       ijson ile streaming — RAM dostu."""
    for fp in file_paths:
        try:
            with gzip.open(fp, 'rb') as gz:
                # Üst seviye array'in her elemanını sırayla parse et
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

def iter_ctu_events(file_paths):
    """CTU Zeek JSON-lines."""
    for fp in file_paths:
        try:
            with gzip.open(fp, 'rt', errors='ignore') as gz:
                for line in gz:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    ip = ev.get('id.orig_h')
                    if ip:
                        yield ip, ev, fp
        except Exception as e:
            print(f"      ⚠ Dosya hatası ({fp}): {e}")


# ══════════════════════════════════════════════════════════════
# YARDIMCI: Streaming sekans inşası
# ══════════════════════════════════════════════════════════════
def build_sequences(event_iter, extractor, valid_ips, seq_len,
                    label_str, total_files):
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
                print(f"      [{label_str}] {file_count}/{total_files} dosya | "
                      f"okunan:{n_total:,} | tutulan:{n_kept:,} | "
                      f"dolu IP: {len(filled)}/{len(valid_ips)} | {el:.1f}s")

        n_total += 1
        if ip not in valid_ips or ip in filled:
            continue
        feats, t = extractor(ev, prev_ts.get(ip, 0.0))
        prev_ts[ip] = t
        sequences.setdefault(ip, []).append(feats)
        n_kept += 1
        if len(sequences[ip]) >= seq_len:
            filled.add(ip)
            if len(filled) == len(valid_ips):
                print(f"      [{label_str}] Tüm IP'ler doldu, erken bitiş.")
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
            seq = pad + seq  # pre-pad
        out[ip] = seq
    return out, n_features


def get_cowrie_files():
    return sorted(glob(os.path.join(COWRIE_RAW_DIR, "cyberlab_*.json.gz")))


def get_ctu_conn_files(honeypots=None, max_days=None):
    if honeypots is None:
        honeypots = sorted([d for d in os.listdir(CTU_ROOT)
                            if d.startswith("Honeypot-")])
    files = []
    for hp in honeypots:
        hp_dir = os.path.join(CTU_ROOT, hp)
        if not os.path.isdir(hp_dir):
            continue
        days = sorted([d for d in os.listdir(hp_dir)
                       if os.path.isdir(os.path.join(hp_dir, d))])
        if max_days is not None:
            days = days[:max_days]
        for day in days:
            day_dir = os.path.join(hp_dir, day)
            conns = sorted(glob(os.path.join(day_dir, "conn.*.log.gz")))
            files.extend(conns)
    return files


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
def build_lstm_model(seq_len, n_features, n_classes, model_type="standard"):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(seq_len, n_features)))
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

    lr = 5e-4 if model_type == "bidirectional" else 1e-3
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=opt, loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model


def train_and_evaluate(X_train, X_test, y_train, y_test, class_names,
                       dataset_name, phase_name):
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = len(class_names)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if n_classes == 2:
        y_train_cat = y_train_enc
        y_test_cat = y_test_enc
    else:
        y_train_cat = to_categorical(y_train_enc, n_classes)
        y_test_cat = to_categorical(y_test_enc, n_classes)

    cw_raw = compute_class_weight('balanced',
                                  classes=np.unique(y_train_enc),
                                  y=y_train_enc)
    cw_smooth = np.sqrt(cw_raw)
    cw_smooth = cw_smooth / cw_smooth.mean()
    class_weight_dict = dict(zip(np.unique(y_train_enc), cw_smooth))
    print(f"    Class weights: "
          f"{ {int(k): round(float(v),3) for k,v in class_weight_dict.items()} }")

    early_stop = EarlyStopping(monitor='val_loss', patience=12,
                               min_delta=1e-4, restore_best_weights=True,
                               verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-6, verbose=0)

    results = {}
    for model_type in ["standard", "bidirectional"]:
        name = "LSTM" if model_type == "standard" else "Bi-LSTM"
        print(f"\n    [{name}] Eğitiliyor...")
        model = build_lstm_model(seq_len, n_features, n_classes, model_type)

        t = time.time()
        hist = model.fit(X_train, y_train_cat,
                         validation_split=0.2, epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         class_weight=class_weight_dict,
                         callbacks=[early_stop, reduce_lr], verbose=0)
        train_t = time.time() - t

        t = time.time()
        if n_classes == 2:
            y_pred = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
        else:
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        pred_t = time.time() - t

        acc = accuracy_score(y_test_enc, y_pred)
        f1 = f1_score(y_test_enc, y_pred, average='weighted')
        prec = precision_score(y_test_enc, y_pred, average='weighted')
        rec = recall_score(y_test_enc, y_pred, average='weighted')

        print(f"      Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Eğitim: {train_t:.1f}s | Tahmin: {pred_t:.3f}s | "
              f"Epoch: {len(hist.history['loss'])}")

        results[name] = dict(accuracy=acc, f1=f1, precision=prec, recall=rec,
                             train_time=train_t, predict_time=pred_t,
                             epochs=len(hist.history['loss']))

        rep = classification_report(y_test_enc, y_pred, target_names=class_names,
                                    output_dict=True)
        pd.DataFrame(rep).transpose().to_csv(
            os.path.join(OUTPUT_DIR, "data",
                         f"{dataset_name}_{model_type}_report.csv"))

        cm = confusion_matrix(y_test_enc, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap='Blues')
        ax.set_title(f"Confusion — {name}\n{phase_name} ({dataset_name})")
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
                                 f"{dataset_name}_{model_type}_confusion.png"),
                    dpi=150)
        plt.close()

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(hist.history['loss'], label='train')
        a1.plot(hist.history['val_loss'], label='val')
        a1.set_title(f"{name} Loss"); a1.legend()
        a2.plot(hist.history['accuracy'], label='train')
        a2.plot(hist.history['val_accuracy'], label='val')
        a2.set_title(f"{name} Acc"); a2.legend()
        plt.suptitle(f"{phase_name} — {dataset_name}", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figures",
                                 f"{dataset_name}_{model_type}_history.png"),
                    dpi=150)
        plt.close()
    return results


# ══════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════
def make_tensor(seq_dict, label_map):
    X, y, ips = [], [], []
    for ip, seq in seq_dict.items():
        if ip not in label_map:
            continue
        X.append(seq)
        y.append(label_map[ip])
        ips.append(ip)
    return np.array(X, dtype=np.float32), np.array(y), ips


print("\n" + "=" * 60)
print("ETİKET YÜKLEME")
print("=" * 60)

cowrie_w6_labels = load_labels(LABELED_W6_COWRIE, ['label', 'threat_label'])
cowrie_w7_labels = load_labels(LABELED_W7_COWRIE,
                               ['threat_label', 'threat_level', 'label'])
ctu_w6_labels = load_labels(LABELED_W6_CTU, ['label', 'behavior_label'])
ctu_w7_labels = load_labels(LABELED_W7_CTU,
                            ['behavior_label', 'behavior_type', 'label'])

cowrie_ips = set(cowrie_w6_labels) | set(cowrie_w7_labels)
ctu_ips = set(ctu_w6_labels) | set(ctu_w7_labels)
print(f"  Cowrie etiketli IP: W6={len(cowrie_w6_labels)} "
      f"W7={len(cowrie_w7_labels)} birleşim={len(cowrie_ips)}")
print(f"  CTU etiketli IP: W6={len(ctu_w6_labels)} "
      f"W7={len(ctu_w7_labels)} birleşim={len(ctu_ips)}")


print("\n" + "=" * 60)
print("COWRIE — SEKANS İNŞASI (streaming)")
print("=" * 60)
cowrie_files = get_cowrie_files()
print(f"  {len(cowrie_files)} dosya bulundu")
cowrie_seq, cowrie_nfeat = build_sequences(
    iter_cowrie_events(cowrie_files), extract_cowrie_features,
    valid_ips=cowrie_ips, seq_len=SEQ_LEN,
    label_str="Cowrie", total_files=len(cowrie_files))
print(f"  Sekans oluşturulan IP: {len(cowrie_seq)} | "
      f"feature/event: {cowrie_nfeat}")


print("\n" + "=" * 60)
print("CTU — SEKANS İNŞASI (streaming)")
print("=" * 60)
ctu_files = get_ctu_conn_files(CTU_HONEYPOTS, CTU_MAX_DAYS)
print(f"  {len(ctu_files)} conn.log dosyası bulundu")
ctu_seq, ctu_nfeat = build_sequences(
    iter_ctu_events(ctu_files), extract_ctu_features,
    valid_ips=ctu_ips, seq_len=SEQ_LEN,
    label_str="CTU", total_files=len(ctu_files))
print(f"  Sekans oluşturulan IP: {len(ctu_seq)} | "
      f"feature/event: {ctu_nfeat}")


# ══════════════════════════════════════════════════════════════
# Z-SCORE normalizasyon (timestep boyunca)
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# 4 SENARYOYU EĞİT
# ══════════════════════════════════════════════════════════════
all_results = {}

scenarios = [
    ("Hafta 6 — Cowrie", "cowrie_w6_seq", cowrie_seq, cowrie_w6_labels),
    ("Hafta 6 — CTU",    "ctu_w6_seq",    ctu_seq,    ctu_w6_labels),
    ("Hafta 7 — Cowrie", "cowrie_w7_seq", cowrie_seq, cowrie_w7_labels),
    ("Hafta 7 — CTU",    "ctu_w7_seq",    ctu_seq,    ctu_w7_labels),
]

for phase, name, seq_dict, label_map in scenarios:
    print("\n" + "=" * 60)
    print(f"{phase}")
    print("=" * 60)
    X, y, ips = make_tensor(seq_dict, label_map)
    if len(X) == 0:
        print(f"  ⚠ Hiç eşleşen IP yok, atlanıyor.")
        all_results[phase] = None
        continue
    classes, counts = np.unique(y, return_counts=True)
    print(f"  Örnek sayısı: {len(X)} | Sınıflar: {dict(zip(classes, counts))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_tr, X_te = normalize_3d(X_tr, X_te)
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

    all_results[phase] = train_and_evaluate(
        X_tr, X_te, y_tr, y_te,
        class_names=sorted(np.unique(y).tolist()),
        dataset_name=name, phase_name=phase)


# ══════════════════════════════════════════════════════════════
# ÖZET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ÖZET — SEKANS LSTM")
print("=" * 60)
print(f"\n  {'Phase':<20} {'Model':<10} {'F1':>8} {'Acc':>8} "
      f"{'Prec':>8} {'Recall':>8} {'Eğitim(s)':>10}")
print("  " + "─" * 74)

rows = []
for phase, res in all_results.items():
    if res is None: continue
    for m, r in res.items():
        print(f"  {phase:<20} {m:<10} {r['f1']:>8.4f} {r['accuracy']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['train_time']:>10.1f}")
        rows.append(dict(Phase=phase, Model=m,
                         F1=round(r['f1'], 4), Accuracy=round(r['accuracy'], 4),
                         Precision=round(r['precision'], 4),
                         Recall=round(r['recall'], 4),
                         Train_Time_s=round(r['train_time'], 1),
                         Epochs=r['epochs']))
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "data",
                                       "lstm_seq_summary.csv"), index=False)
print(f"\n  Çıktılar: {OUTPUT_DIR}")
print("  ✓ Sekans LSTM tamamlandı.")