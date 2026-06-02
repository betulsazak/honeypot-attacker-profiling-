"""
Hafta 4: Kimlik Bilgisi Analizi (v7 — Bellek-Dostu)
=====================================================
Cowrie parquet dosyasından kimlik bilgisi saldırı analizi.

Analizler & Çıktılar:
  ═══════════════════════════════════════════════════════════
  A. KOMBİNASYON ANALİZİ
  ═══════════════════════════════════════════════════════════
    A1. Top-30 kullanıcı adı (login.success + login.failed)
    A2. Top-30 şifre
    A3. Top-30 kullanıcı:şifre kombinasyonu
    A4. Başarılı vs başarısız giriş karşılaştırması
    A5. Kullanıcı adı ve şifre örtüşme analizi (aynı string
        hem kullanıcı hem şifre olarak denenmiş mi?)

  ═══════════════════════════════════════════════════════════
  B. ŞİFRE ENTROPİ VE KARMAŞIKLİK ANALİZİ
  ═══════════════════════════════════════════════════════════
    B1. Shannon entropi dağılımı (histogram)
    B2. Şifre uzunluk dağılımı
    B3. Karmaşıklık sınıflandırması:
        - Sadece sayısal (123456)
        - Sadece küçük harf (password)
        - Alfanümerik (admin123)
        - Karışık (büyük+küçük+sayı)
        - Özel karakterli (P@ssw0rd!)
    B4. Entropi vs uzunluk scatter plot
    B5. Entropi kategorileri (çok zayıf / zayıf / orta / güçlü)

  ═══════════════════════════════════════════════════════════
  C. ZAMANSAL & COĞRAFİ KREDENSİYEL PATERNLERİ
  ═══════════════════════════════════════════════════════════
    C1. Saatlik giriş denemesi yoğunluğu (success vs failed)
    C2. Top-10 ülke × en çok denenen şifre
    C3. Günlük benzersiz şifre sayısı trendi
    C4. Saldırgan IP başına ortalama deneme sayısı dağılımı

  ═══════════════════════════════════════════════════════════
  D. SALDIRGAN DAVRANIŞ PROFİLLERİ
  ═══════════════════════════════════════════════════════════
    D1. Sözlük saldırısı tespiti (IP başına benzersiz
        kullanıcı/şifre çeşitliliği)
    D2. Brute-force vs dictionary attack sınıflandırması
    D3. Credential stuffing göstergeleri (aynı kombinasyonun
        farklı IP'lerden denenmesi)

  Tüm çıktılar → OUTPUT_DIR/week4/
"""

import os
import gc
import math
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# YAPILANDIRMA
# =====================================================================

OUTPUT_DIR  = "/mnt/c/TEZ_PROJEEE/output"
WEEK4_DIR   = os.path.join(OUTPUT_DIR, "week4")
WEEK4_FIGS  = os.path.join(WEEK4_DIR, "figures")
WEEK4_DATA  = os.path.join(WEEK4_DIR, "data")

COWRIE_PATH = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")

BATCH_SIZE  = 500_000

# Matplotlib
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

C_SUCCESS = "#27AE60"
C_FAILED  = "#E74C3C"
C_PRIMARY = "#A23B72"
C_ACCENT  = "#F18F01"
C_DARK    = "#2C3E50"


# =====================================================================
# YARDIMCI — Sütun kontrolü & chunk okuma
# =====================================================================

def _get_columns(parquet_path):
    schema = pq.read_schema(parquet_path)
    return [schema.field(i).name for i in range(len(schema))]


def _fmt(x, _=None):
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# =====================================================================
# ŞİFRE ENTROPİ & KARMAŞIKLİK FONKSİYONLARI
# =====================================================================

def shannon_entropy(s):
    """Bir string'in Shannon entropisini hesaplar (bit cinsinden)."""
    if not s or len(s) == 0:
        return 0.0
    freq = Counter(s)
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def classify_complexity(password):
    """
    Şifre karmaşıklık sınıflandırması.
    Döndürür: kategori string'i
    """
    if not password or len(password) == 0:
        return "Boş"

    has_lower   = any(c.islower() for c in password)
    has_upper   = any(c.isupper() for c in password)
    has_digit   = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)

    if password.isdigit():
        return "Sadece Sayısal"
    if password.isalpha() and password.islower():
        return "Sadece Küçük Harf"
    if password.isalpha() and password.isupper():
        return "Sadece Büyük Harf"
    if password.isalpha():
        return "Karışık Harf"
    if has_special and has_digit and (has_lower or has_upper):
        return "Özel Karakterli"
    if has_lower and has_upper and has_digit:
        return "Alfanümerik Karışık"
    if (has_lower or has_upper) and has_digit:
        return "Alfanümerik"
    if has_special:
        return "Özel Karakterli"
    return "Diğer"


def entropy_category(entropy):
    """Entropi değerini güvenlik kategorisine çevirir."""
    if entropy < 2.0:
        return "Çok Zayıf"
    elif entropy < 3.0:
        return "Zayıf"
    elif entropy < 3.5:
        return "Orta"
    else:
        return "Güçlü"


# =====================================================================
# VERİ TOPLAMA — Chunk-chunk
# =====================================================================

def collect_login_data(cowrie_path):
    """
    Cowrie parquet'ten login eventlerini chunk-chunk okur.
    Döndürür: {
        'user_counts': Counter,    # kullanıcı adı → toplam deneme
        'pass_counts': Counter,    # şifre → toplam deneme
        'combo_counts': Counter,   # (user, pass) → toplam deneme
        'success_users': Counter,  # başarılı giriş kullanıcı adları
        'success_passes': Counter, # başarılı giriş şifreleri
        'failed_users': Counter,   # başarısız giriş kullanıcı adları
        'failed_passes': Counter,  # başarısız giriş şifreleri
        'total_success': int,
        'total_failed': int,
    }
    """
    print("  Login verileri toplanıyor (chunk-chunk)...")

    cols = _get_columns(cowrie_path)
    has_user = "username" in cols
    has_pass = "password" in cols
    has_event = "eventid" in cols

    if not has_user and not has_pass:
        print("    UYARI: username/password sütunu bulunamadı")
        return None

    read_cols = ["eventid"]
    if has_user: read_cols.append("username")
    if has_pass: read_cols.append("password")

    data = {
        "user_counts": Counter(),
        "pass_counts": Counter(),
        "combo_counts": Counter(),
        "success_users": Counter(),
        "success_passes": Counter(),
        "failed_users": Counter(),
        "failed_passes": Counter(),
        "total_success": 0,
        "total_failed": 0,
    }

    pf = pq.ParquetFile(cowrie_path)
    processed = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        df = batch.to_pandas()
        # Login eventlerini filtrele
        login_mask = df["eventid"].astype(str).str.contains("login", na=False)
        logins = df[login_mask].copy()
        del df

        if len(logins) == 0:
            gc.collect()
            continue

        success_mask = logins["eventid"].astype(str).str.contains("success", na=False)
        failed_mask = ~success_mask

        data["total_success"] += int(success_mask.sum())
        data["total_failed"] += int(failed_mask.sum())

        if has_user:
            users = logins["username"].dropna()
            data["user_counts"].update(users.value_counts().to_dict())
            data["success_users"].update(
                logins.loc[success_mask, "username"].dropna().value_counts().to_dict()
            )
            data["failed_users"].update(
                logins.loc[failed_mask, "username"].dropna().value_counts().to_dict()
            )

        if has_pass:
            passes = logins["password"].dropna()
            data["pass_counts"].update(passes.value_counts().to_dict())
            data["success_passes"].update(
                logins.loc[success_mask, "password"].dropna().value_counts().to_dict()
            )
            data["failed_passes"].update(
                logins.loc[failed_mask, "password"].dropna().value_counts().to_dict()
            )

        if has_user and has_pass:
            combos = logins[["username", "password"]].dropna()
            if len(combos) > 0:
                combo_strs = combos["username"].astype(str) + ":::" + combos["password"].astype(str)
                data["combo_counts"].update(combo_strs.value_counts().to_dict())

        processed += len(logins)
        del logins
        gc.collect()

        if processed % 2_000_000 < BATCH_SIZE:
            print(f"    ... {processed:,} login kaydı işlendi")

    print(f"    Toplam: {processed:,} login ({data['total_success']:,} başarılı, {data['total_failed']:,} başarısız)")
    return data


def collect_password_entropy_data(cowrie_path, sample_max=500_000):
    """Benzersiz şifrelerin entropi ve karmaşıklık bilgilerini toplar."""
    print("  Şifre entropi verileri toplanıyor...")

    cols = _get_columns(cowrie_path)
    if "password" not in cols:
        return None

    # Benzersiz şifreler ve frekansları
    pass_counts = Counter()
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["eventid", "password"]):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)]
        pass_counts.update(logins["password"].dropna().value_counts().to_dict())
        del df, logins; gc.collect()

    print(f"    {len(pass_counts):,} benzersiz şifre bulundu")

    # Entropi ve karmaşıklık hesapla
    records = []
    for pwd, count in pass_counts.most_common(sample_max):
        pwd_str = str(pwd)
        records.append({
            "password": pwd_str,
            "count": count,
            "length": len(pwd_str),
            "entropy": shannon_entropy(pwd_str),
            "complexity": classify_complexity(pwd_str),
            "entropy_cat": entropy_category(shannon_entropy(pwd_str)),
        })

    del pass_counts; gc.collect()
    df = pd.DataFrame(records)
    print(f"    {len(df):,} şifre analiz edildi")
    return df


def collect_temporal_credential_data(cowrie_path):
    """Saatlik ve günlük credential denemesi verileri."""
    print("  Zamansal credential verileri toplanıyor...")

    cols = _get_columns(cowrie_path)
    read_cols = ["eventid", "hour", "date"]
    if "src_ip" in cols: read_cols.append("src_ip")
    if "password" in cols: read_cols.append("password")
    if "username" in cols: read_cols.append("username")

    hourly_success = Counter()
    hourly_failed = Counter()
    daily_unique_pass = {}
    ip_attempt_count = Counter()

    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        df = batch.to_pandas()
        login_mask = df["eventid"].astype(str).str.contains("login", na=False)
        logins = df[login_mask].copy()
        del df

        if len(logins) == 0:
            gc.collect()
            continue

        success = logins["eventid"].astype(str).str.contains("success", na=False)

        # Saatlik
        for h, cnt in logins.loc[success, "hour"].dropna().astype(int).value_counts().items():
            hourly_success[h] += cnt
        for h, cnt in logins.loc[~success, "hour"].dropna().astype(int).value_counts().items():
            hourly_failed[h] += cnt

        # Günlük benzersiz şifre — nunique ile say (set tutmak yerine)
        if "password" in logins.columns:
            valid_dp = logins[["date", "password"]].dropna()
            if len(valid_dp) > 0:
                dn = valid_dp.groupby("date")["password"].nunique()
                for dt, cnt in dn.items():
                    daily_unique_pass[dt] = daily_unique_pass.get(dt, 0) + cnt
            del valid_dp

        # IP başına deneme
        if "src_ip" in logins.columns:
            ip_attempt_count.update(logins["src_ip"].dropna().value_counts().to_dict())

        del logins; gc.collect()

    # daily_unique_pass zaten {date: count} formatında
    daily_unique_counts = dict(daily_unique_pass)
    del daily_unique_pass; gc.collect()

    return {
        "hourly_success": dict(hourly_success),
        "hourly_failed": dict(hourly_failed),
        "daily_unique_pass": daily_unique_counts,
        "ip_attempt_count": dict(ip_attempt_count),
    }


def collect_attacker_behavior(cowrie_path):
    """
    Saldırgan davranış profilleri — IP başına:
    - Benzersiz kullanıcı adı sayısı
    - Benzersiz şifre sayısı
    - Toplam deneme sayısı
    - Benzersiz kombinasyon sayısı

    İki geçişli yaklaşım:
      1. Geçiş: IP başına toplam deneme sayısını topla, top-50K IP'yi belirle
      2. Geçiş: Sadece top IP'ler için benzersiz user/pass/combo setlerini topla
    """
    print("  Saldırgan davranış profilleri toplanıyor...")

    cols = _get_columns(cowrie_path)
    read_cols = ["eventid", "src_ip"]
    has_user = "username" in cols
    has_pass = "password" in cols
    if has_user: read_cols.append("username")
    if has_pass: read_cols.append("password")

    # ── Geçiş 1: IP başına toplam deneme sayısı ──
    print("    Geçiş 1: IP başına deneme sayısı...")
    ip_total = Counter()
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["eventid", "src_ip"]):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)]
        ip_total.update(logins["src_ip"].dropna().value_counts().to_dict())
        del df, logins; gc.collect()

    total_ips = len(ip_total)
    print(f"    {total_ips:,} benzersiz IP bulundu")

    # Analiz için top 50K IP (bellek sınırı)
    MAX_IPS = 50_000
    if total_ips > MAX_IPS:
        top_ip_set = set(ip for ip, _ in ip_total.most_common(MAX_IPS))
        print(f"    Bellek koruması: En aktif {MAX_IPS:,} IP analiz edilecek")
    else:
        top_ip_set = set(ip_total.keys())

    # ── Geçiş 2: Sadece seçili IP'ler için benzersiz sayılar ──
    print("    Geçiş 2: Benzersiz user/pass/combo sayımı...")
    ip_users = defaultdict(set)
    ip_passes = defaultdict(set)
    ip_combos = defaultdict(set)

    pf = pq.ParquetFile(cowrie_path)
    batch_num = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)].copy()
        del df

        if len(logins) == 0:
            gc.collect(); continue

        # Sadece seçili IP'leri tut
        logins = logins[logins["src_ip"].isin(top_ip_set)]
        if len(logins) == 0:
            gc.collect(); continue

        # Vektörel groupby ile benzersiz değerleri setlere ekle
        if has_user:
            valid = logins[["src_ip", "username"]].dropna()
            for ip, grp in valid.groupby("src_ip")["username"]:
                ip_users[ip].update(grp.unique())
            del valid

        if has_pass:
            valid = logins[["src_ip", "password"]].dropna()
            for ip, grp in valid.groupby("src_ip")["password"]:
                ip_passes[ip].update(grp.unique())
            del valid

        if has_user and has_pass:
            valid = logins[["src_ip", "username", "password"]].dropna()
            if len(valid) > 0:
                valid["combo"] = valid["username"].astype(str) + ":::" + valid["password"].astype(str)
                for ip, grp in valid.groupby("src_ip")["combo"]:
                    ip_combos[ip].update(grp.unique())
            del valid

        del logins; gc.collect()
        batch_num += 1
        if batch_num % 20 == 0:
            print(f"    ... batch {batch_num}")

    # DataFrame oluştur
    records = []
    for ip in top_ip_set:
        records.append({
            "src_ip": ip,
            "total_attempts": ip_total.get(ip, 0),
            "unique_users": len(ip_users.get(ip, set())),
            "unique_passes": len(ip_passes.get(ip, set())),
            "unique_combos": len(ip_combos.get(ip, set())),
        })

    del ip_users, ip_passes, ip_combos, ip_total, top_ip_set
    gc.collect()

    df = pd.DataFrame(records)
    print(f"    {len(df):,} saldırgan IP analiz edildi")
    return df


def collect_credential_stuffing(cowrie_path, top_n=50):
    """
    Credential stuffing: aynı (user, pass) çiftini kaç farklı IP denemiş?
    İki geçişli: önce top combo'ları bul, sonra sadece onlar için IP setlerini topla.
    """
    print("  Credential stuffing analizi...")

    cols = _get_columns(cowrie_path)
    if "username" not in cols or "password" not in cols or "src_ip" not in cols:
        return None

    # ── Geçiş 1: En çok denenen 500 kombinasyonu bul ──
    print("    Geçiş 1: Top kombinasyonlar...")
    combo_total = Counter()
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                  columns=["eventid", "username", "password"]):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)]
        valid = logins[["username", "password"]].dropna()
        if len(valid) > 0:
            valid["combo"] = valid["username"].astype(str) + ":::" + valid["password"].astype(str)
            combo_total.update(valid["combo"].value_counts().to_dict())
        del df, logins, valid; gc.collect()

    top_combos = set(c for c, _ in combo_total.most_common(500))
    print(f"    {len(combo_total):,} benzersiz combo, top 500 seçildi")

    # ── Geçiş 2: Sadece top combo'lar için IP setlerini topla ──
    print("    Geçiş 2: IP dağılımı...")
    combo_ips = defaultdict(set)
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                  columns=["eventid", "src_ip", "username", "password"]):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)]
        valid = logins[["src_ip", "username", "password"]].dropna()
        del df, logins

        if len(valid) == 0:
            gc.collect(); continue

        valid["combo"] = valid["username"].astype(str) + ":::" + valid["password"].astype(str)
        valid = valid[valid["combo"].isin(top_combos)]

        for combo_str, grp in valid.groupby("combo")["src_ip"]:
            combo_ips[combo_str].update(grp.unique())

        del valid; gc.collect()

    # Sonuç
    records = []
    for combo_str in sorted(combo_ips, key=lambda x: len(combo_ips[x]), reverse=True)[:top_n]:
        parts = combo_str.split(":::", 1)
        if len(parts) == 2:
            records.append({
                "username": parts[0],
                "password": parts[1],
                "unique_ips": len(combo_ips[combo_str]),
                "total_attempts": combo_total.get(combo_str, 0),
            })

    del combo_ips, combo_total, top_combos; gc.collect()
    print(f"    Top-{top_n} credential stuffing kombinasyonu hazır")
    return pd.DataFrame(records)


def collect_country_passwords(cowrie_path, top_countries=10, top_passwords=10):
    """Top-N ülke × top-N şifre çapraz tablosu."""
    print("  Ülke × şifre çapraz analizi...")

    cols = _get_columns(cowrie_path)
    cc_col = next((c for c in ["geo_country_code", "src_country_code"] if c in cols), None)
    if not cc_col or "password" not in cols:
        return None

    # Önce top ülkeleri bul
    country_counts = Counter()
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["eventid", cc_col]):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)]
        country_counts.update(logins[cc_col].dropna().value_counts().to_dict())
        del df, logins; gc.collect()

    top_cc = [c for c, _ in country_counts.most_common(top_countries)]
    del country_counts; gc.collect()

    # Ülke × şifre sayımı — vektörel
    country_pass = defaultdict(Counter)
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                  columns=["eventid", cc_col, "password"]):
        df = batch.to_pandas()
        logins = df[df["eventid"].astype(str).str.contains("login", na=False)]
        logins = logins[logins[cc_col].isin(top_cc)][[cc_col, "password"]].dropna()
        del df

        if len(logins) == 0:
            gc.collect()
            continue

        for cc, grp in logins.groupby(cc_col):
            country_pass[cc].update(grp["password"].value_counts().to_dict())

        del logins; gc.collect()

    # Her ülke için top-N şifre
    result = {}
    for cc in top_cc:
        result[cc] = dict(country_pass[cc].most_common(top_passwords))

    del country_pass; gc.collect()
    return result


# =====================================================================
# A. KOMBİNASYON ANALİZİ GRAFİKLERİ
# =====================================================================

def fig_A1_top_usernames(login_data, save_dir):
    print("\n  [A1] Top-30 kullanıcı adı...")
    top = login_data["user_counts"].most_common(30)
    if not top: print("    Veri yok — atlanıyor."); return

    labels, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(range(len(labels)), vals, color=C_PRIMARY, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Deneme Sayısı")
    ax.set_title("En Çok Denenen 30 Kullanıcı Adı")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                _fmt(val), va="center", fontsize=7, color="#333")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "A1_top30_usernames.png")); plt.close(fig)
    print(f"    → A1_top30_usernames.png")


def fig_A2_top_passwords(login_data, save_dir):
    print("\n  [A2] Top-30 şifre...")
    top = login_data["pass_counts"].most_common(30)
    if not top: print("    Veri yok — atlanıyor."); return

    labels, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(range(len(labels)), vals, color=C_ACCENT, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Deneme Sayısı")
    ax.set_title("En Çok Denenen 30 Şifre")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                _fmt(val), va="center", fontsize=7, color="#333")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "A2_top30_passwords.png")); plt.close(fig)
    print(f"    → A2_top30_passwords.png")


def fig_A3_top_combos(login_data, save_dir):
    print("\n  [A3] Top-30 kullanıcı:şifre kombinasyonu...")
    top = login_data["combo_counts"].most_common(30)
    if not top: print("    Veri yok — atlanıyor."); return

    labels = []
    vals = []
    for combo_key, count in top:
        if isinstance(combo_key, tuple):
            labels.append(f"{combo_key[0]}:{combo_key[1]}")
        else:
            labels.append(combo_key.replace(":::", ":"))
        vals.append(count)

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(range(len(labels)), vals, color=C_DARK, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Deneme Sayısı")
    ax.set_title("En Çok Denenen 30 Kullanıcı:Şifre Kombinasyonu")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                _fmt(val), va="center", fontsize=7, color="#333")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "A3_top30_combos.png")); plt.close(fig)
    print(f"    → A3_top30_combos.png")


def fig_A4_success_vs_failed(login_data, save_dir):
    print("\n  [A4] Başarılı vs başarısız giriş karşılaştırması...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Pasta
    s, f = login_data["total_success"], login_data["total_failed"]
    axes[0].pie([s, f], labels=["Başarılı", "Başarısız"],
                autopct="%1.2f%%", colors=[C_SUCCESS, C_FAILED],
                startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Genel Oran")

    # Top-10 başarılı kullanıcı
    top_s = login_data["success_users"].most_common(10)
    if top_s:
        labels_s, vals_s = zip(*top_s)
        axes[1].barh(range(len(labels_s)), vals_s, color=C_SUCCESS, alpha=0.85)
        axes[1].set_yticks(range(len(labels_s)))
        axes[1].set_yticklabels(labels_s, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Giriş Sayısı")
        axes[1].set_title("Top-10 Başarılı Kullanıcı")
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))

    # Top-10 başarılı şifre
    top_sp = login_data["success_passes"].most_common(10)
    if top_sp:
        labels_sp, vals_sp = zip(*top_sp)
        axes[2].barh(range(len(labels_sp)), vals_sp, color=C_SUCCESS, alpha=0.85)
        axes[2].set_yticks(range(len(labels_sp)))
        axes[2].set_yticklabels(labels_sp, fontsize=9)
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Giriş Sayısı")
        axes[2].set_title("Top-10 Başarılı Şifre")
        axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))

    plt.suptitle("Başarılı vs Başarısız Giriş Denemeleri", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "A4_success_vs_failed.png")); plt.close(fig)
    print(f"    → A4_success_vs_failed.png")


def fig_A5_overlap(login_data, save_dir):
    print("\n  [A5] Kullanıcı-şifre örtüşme analizi...")

    user_set = set(login_data["user_counts"].keys())
    pass_set = set(login_data["pass_counts"].keys())
    overlap = user_set & pass_set

    fig, ax = plt.subplots(figsize=(8, 6))

    # Basit Venn benzeri gösterim
    from matplotlib.patches import Circle

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    c1 = Circle((-0.7, 0), 1.5, fill=True, facecolor=C_PRIMARY, alpha=0.3, edgecolor=C_PRIMARY, linewidth=2)
    c2 = Circle((0.7, 0), 1.5, fill=True, facecolor=C_ACCENT, alpha=0.3, edgecolor=C_ACCENT, linewidth=2)
    ax.add_patch(c1)
    ax.add_patch(c2)

    ax.text(-1.6, 0, f"Sadece\nKullanıcı\n{len(user_set - overlap):,}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0, 0, f"Ortak\n{len(overlap):,}", ha="center", va="center", fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(1.6, 0, f"Sadece\nŞifre\n{len(pass_set - overlap):,}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(-1.6, -1.7, "Kullanıcı Adları", ha="center", fontsize=10, color=C_PRIMARY)
    ax.text(1.6, -1.7, "Şifreler", ha="center", fontsize=10, color=C_ACCENT)

    ax.set_title("Kullanıcı Adı ve Şifre Olarak Ortak Kullanılan Stringler", fontsize=13, fontweight="bold")

    # Örtüşen top örnekler
    if overlap:
        top_overlap = sorted(overlap, key=lambda x: login_data["user_counts"].get(x, 0) + login_data["pass_counts"].get(x, 0), reverse=True)[:10]
        text = "Örnekler: " + ", ".join(top_overlap)
        ax.text(0, -1.9, text, ha="center", fontsize=8, style="italic", color="#555")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "A5_user_pass_overlap.png")); plt.close(fig)
    print(f"    → A5_user_pass_overlap.png  ({len(overlap):,} ortak string)")


# =====================================================================
# B. ŞİFRE ENTROPİ GRAFİKLERİ
# =====================================================================

def fig_B1_entropy_histogram(entropy_df, save_dir):
    print("\n  [B1] Shannon entropi dağılımı...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Benzersiz şifre bazlı
    axes[0].hist(entropy_df["entropy"], bins=50, color=C_PRIMARY, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Shannon Entropi (bit)")
    axes[0].set_ylabel("Benzersiz Şifre Sayısı")
    axes[0].set_title("Entropi Dağılımı (Benzersiz Şifreler)")
    axes[0].axvline(x=2.0, color="red", linestyle="--", alpha=0.7, label="Çok Zayıf (<2)")
    axes[0].axvline(x=3.0, color="orange", linestyle="--", alpha=0.7, label="Zayıf (<3)")
    axes[0].axvline(x=3.5, color="green", linestyle="--", alpha=0.7, label="Orta (<3.5)")
    axes[0].legend(fontsize=8)

    # Frekans ağırlıklı
    axes[1].hist(entropy_df["entropy"], bins=50, weights=entropy_df["count"],
                 color=C_ACCENT, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Shannon Entropi (bit)")
    axes[1].set_ylabel("Toplam Deneme Sayısı")
    axes[1].set_title("Entropi Dağılımı (Frekans Ağırlıklı)")
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    axes[1].axvline(x=2.0, color="red", linestyle="--", alpha=0.7)
    axes[1].axvline(x=3.0, color="orange", linestyle="--", alpha=0.7)
    axes[1].axvline(x=3.5, color="green", linestyle="--", alpha=0.7)

    plt.suptitle("Şifre Entropi Dağılımı", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "B1_entropy_histogram.png")); plt.close(fig)
    print(f"    → B1_entropy_histogram.png")


def fig_B2_length_distribution(entropy_df, save_dir):
    print("\n  [B2] Şifre uzunluk dağılımı...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_len = min(entropy_df["length"].max(), 40)
    bins = range(0, max_len + 2)

    axes[0].hist(entropy_df["length"], bins=bins, color=C_PRIMARY, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Şifre Uzunluğu")
    axes[0].set_ylabel("Benzersiz Şifre Sayısı")
    axes[0].set_title("Uzunluk Dağılımı (Benzersiz)")

    axes[1].hist(entropy_df["length"], bins=bins, weights=entropy_df["count"],
                 color=C_ACCENT, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Şifre Uzunluğu")
    axes[1].set_ylabel("Toplam Deneme Sayısı")
    axes[1].set_title("Uzunluk Dağılımı (Ağırlıklı)")
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))

    plt.suptitle("Şifre Uzunluk Dağılımı", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "B2_length_distribution.png")); plt.close(fig)
    print(f"    → B2_length_distribution.png")


def fig_B3_complexity_classes(entropy_df, save_dir):
    print("\n  [B3] Şifre karmaşıklık sınıflandırması...")

    # Benzersiz şifre bazlı
    unique_counts = entropy_df["complexity"].value_counts()
    # Frekans ağırlıklı
    weighted = entropy_df.groupby("complexity")["count"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_counts)))

    axes[0].pie(unique_counts.values, labels=unique_counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontsize": 9})
    axes[0].set_title("Benzersiz Şifre Bazlı")

    axes[1].pie(weighted.values, labels=weighted.index, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Frekans Ağırlıklı")

    plt.suptitle("Şifre Karmaşıklık Sınıflandırması", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "B3_complexity_classes.png")); plt.close(fig)
    print(f"    → B3_complexity_classes.png")


def fig_B4_entropy_vs_length(entropy_df, save_dir):
    print("\n  [B4] Entropi vs uzunluk scatter plot...")

    # Çok fazla noktayı önlemek için örnekle
    if len(entropy_df) > 10_000:
        sample = entropy_df.sample(10_000, random_state=42)
    else:
        sample = entropy_df

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        sample["length"], sample["entropy"],
        c=np.log10(sample["count"] + 1),
        cmap="YlOrRd", alpha=0.5, s=10, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("log₁₀(Deneme Sayısı)")

    ax.set_xlabel("Şifre Uzunluğu")
    ax.set_ylabel("Shannon Entropi (bit)")
    ax.set_title("Şifre Entropi vs Uzunluk İlişkisi")

    # Referans çizgileri
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(y=3.0, color="orange", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(y=3.5, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(ax.get_xlim()[1]*0.95, 1.8, "Çok Zayıf", ha="right", fontsize=8, color="red")
    ax.text(ax.get_xlim()[1]*0.95, 2.8, "Zayıf", ha="right", fontsize=8, color="orange")
    ax.text(ax.get_xlim()[1]*0.95, 3.7, "Güçlü", ha="right", fontsize=8, color="green")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "B4_entropy_vs_length.png")); plt.close(fig)
    print(f"    → B4_entropy_vs_length.png")


def fig_B5_entropy_categories(entropy_df, save_dir):
    print("\n  [B5] Entropi kategorileri...")

    cat_order = ["Çok Zayıf", "Zayıf", "Orta", "Güçlü"]
    cat_colors = ["#E74C3C", "#F39C12", "#F1C40F", "#27AE60"]

    unique_cats = entropy_df["entropy_cat"].value_counts().reindex(cat_order, fill_value=0)
    weighted_cats = entropy_df.groupby("entropy_cat")["count"].sum().reindex(cat_order, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(cat_order, unique_cats.values, color=cat_colors, alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Benzersiz Şifre Sayısı")
    axes[0].set_title("Benzersiz Şifre Bazlı")
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for i, v in enumerate(unique_cats.values):
        axes[0].text(i, v + max(unique_cats.values)*0.02, _fmt(v), ha="center", fontsize=9)

    axes[1].bar(cat_order, weighted_cats.values, color=cat_colors, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Toplam Deneme Sayısı")
    axes[1].set_title("Frekans Ağırlıklı")
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    for i, v in enumerate(weighted_cats.values):
        axes[1].text(i, v + max(weighted_cats.values)*0.02, _fmt(v), ha="center", fontsize=9)

    plt.suptitle("Şifre Entropi Güvenlik Kategorileri", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "B5_entropy_categories.png")); plt.close(fig)
    print(f"    → B5_entropy_categories.png")


# =====================================================================
# C. ZAMANSAL & COĞRAFİ GRAFİKLER
# =====================================================================

def fig_C1_hourly_login(temporal_data, save_dir):
    print("\n  [C1] Saatlik giriş denemesi yoğunluğu...")

    hours = list(range(24))
    success = [temporal_data["hourly_success"].get(h, 0) for h in hours]
    failed = [temporal_data["hourly_failed"].get(h, 0) for h in hours]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    w = 0.4
    ax.bar(x - w/2, failed, w, label="Başarısız", color=C_FAILED, alpha=0.85)
    ax.bar(x + w/2, success, w, label="Başarılı", color=C_SUCCESS, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in hours])
    ax.set_xlabel("Saat (UTC)")
    ax.set_ylabel("Deneme Sayısı")
    ax.set_title("Saatlik Giriş Denemesi Dağılımı (Başarılı vs Başarısız)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "C1_hourly_login_attempts.png")); plt.close(fig)
    print(f"    → C1_hourly_login_attempts.png")


def fig_C2_country_passwords(country_pass_data, save_dir):
    print("\n  [C2] Top-10 ülke × en çok denenen şifre...")

    if not country_pass_data:
        print("    Veri yok — atlanıyor.")
        return

    countries = list(country_pass_data.keys())
    n_countries = len(countries)

    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, cc in enumerate(countries[:10]):
        ax = axes[i]
        top_p = country_pass_data[cc]
        if not top_p:
            ax.text(0.5, 0.5, "Veri yok", transform=ax.transAxes, ha="center")
            ax.set_title(cc)
            continue

        labels = list(top_p.keys())[:10]
        vals = list(top_p.values())[:10]

        ax.barh(range(len(labels)), vals, color=colors[i], alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7, family="monospace")
        ax.invert_yaxis()
        ax.set_title(cc, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
        ax.tick_params(axis="x", labelsize=7)

    # Boş panelleri gizle
    for i in range(n_countries, 10):
        axes[i].axis("off")

    plt.suptitle("Top-10 Ülke — En Çok Denenen Şifreler", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "C2_country_top_passwords.png")); plt.close(fig)
    print(f"    → C2_country_top_passwords.png")


def fig_C3_daily_unique_passwords(temporal_data, save_dir):
    print("\n  [C3] Günlük benzersiz şifre sayısı trendi...")

    daily = temporal_data["daily_unique_pass"]
    if not daily:
        print("    Veri yok — atlanıyor.")
        return

    dates = sorted(daily.keys())
    vals = [daily[d] for d in dates]
    series = pd.Series(vals, index=pd.to_datetime(dates))
    ma7 = series.rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(series.index, series.values, color=C_PRIMARY, alpha=0.3, width=1, label="Günlük")
    ax.plot(ma7.index, ma7.values, color=C_PRIMARY, linewidth=2, label="7-gün HO")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Benzersiz Şifre Sayısı")
    ax.set_title("Günlük Benzersiz Şifre Çeşitliliği")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "C3_daily_unique_passwords.png")); plt.close(fig)
    print(f"    → C3_daily_unique_passwords.png")


def fig_C4_ip_attempt_distribution(temporal_data, save_dir):
    print("\n  [C4] IP başına deneme sayısı dağılımı...")

    ip_counts = temporal_data["ip_attempt_count"]
    if not ip_counts:
        print("    Veri yok — atlanıyor.")
        return

    vals = list(ip_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram (log ölçek)
    axes[0].hist(vals, bins=100, color=C_PRIMARY, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Deneme Sayısı (IP başına)")
    axes[0].set_ylabel("IP Sayısı")
    axes[0].set_title("Doğrusal Ölçek")
    axes[0].set_yscale("log")

    # CDF
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf, color=C_DARK, linewidth=1.5)
    axes[1].set_xlabel("Deneme Sayısı (IP başına)")
    axes[1].set_ylabel("Kümülatif Oran")
    axes[1].set_title("Kümülatif Dağılım (CDF)")
    axes[1].set_xscale("log")

    # İstatistikler
    med = np.median(vals)
    p95 = np.percentile(vals, 95)
    axes[1].axvline(med, color="blue", linestyle="--", alpha=0.7, label=f"Medyan: {med:.0f}")
    axes[1].axvline(p95, color="red", linestyle="--", alpha=0.7, label=f"P95: {p95:.0f}")
    axes[1].legend(fontsize=9)

    plt.suptitle("Saldırgan IP Başına Giriş Denemesi Dağılımı", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "C4_ip_attempt_distribution.png")); plt.close(fig)
    print(f"    → C4_ip_attempt_distribution.png")


# =====================================================================
# D. SALDIRGAN DAVRANIŞ PROFİLLERİ GRAFİKLERİ
# =====================================================================

def fig_D1_attacker_diversity(behavior_df, save_dir):
    print("\n  [D1] Saldırgan çeşitlilik profilleri...")

    if behavior_df is None or len(behavior_df) == 0:
        print("    Veri yok — atlanıyor."); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Benzersiz kullanıcı vs benzersiz şifre scatter
    sample = behavior_df.sample(min(20_000, len(behavior_df)), random_state=42) if len(behavior_df) > 20_000 else behavior_df

    axes[0].scatter(
        sample["unique_users"], sample["unique_passes"],
        c=np.log10(sample["total_attempts"] + 1),
        cmap="YlOrRd", alpha=0.3, s=5, edgecolors="none"
    )
    axes[0].set_xlabel("Benzersiz Kullanıcı Adı")
    axes[0].set_ylabel("Benzersiz Şifre")
    axes[0].set_title("Saldırgan Çeşitlilik (IP Başına)")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")

    # Deneme sayısı vs benzersiz kombinasyon
    axes[1].scatter(
        sample["total_attempts"], sample["unique_combos"],
        c=np.log10(sample["unique_users"] + 1),
        cmap="viridis", alpha=0.3, s=5, edgecolors="none"
    )
    axes[1].set_xlabel("Toplam Deneme")
    axes[1].set_ylabel("Benzersiz Kombinasyon")
    axes[1].set_title("Deneme vs Kombinasyon Çeşitliliği")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")

    plt.suptitle("Saldırgan Davranış Çeşitlilik Profilleri", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "D1_attacker_diversity.png")); plt.close(fig)
    print(f"    → D1_attacker_diversity.png")


def fig_D2_attack_classification(behavior_df, save_dir):
    print("\n  [D2] Brute-force vs dictionary attack sınıflandırması...")

    if behavior_df is None or len(behavior_df) == 0:
        print("    Veri yok — atlanıyor."); return

    # Sınıflandırma kuralları:
    # - Brute-force: az kullanıcı (≤3), çok şifre denemesi
    # - Dictionary: çok kullanıcı (>3), çok şifre
    # - Credential stuffing: çok kombinasyon, az tekrar (combo ≈ attempts)
    # - Hedefli: az kullanıcı, az şifre, çok tekrar

    def classify_attack(row):
        u, p, t, c = row["unique_users"], row["unique_passes"], row["total_attempts"], row["unique_combos"]
        if t <= 3:
            return "Düşük Aktivite"
        if u <= 3 and p > 10:
            return "Brute-force"
        if u > 3 and p > 10:
            return "Dictionary"
        if c > 0 and t / max(c, 1) < 2:
            return "Credential Stuffing"
        return "Hedefli"

    behavior_df = behavior_df.copy()
    behavior_df["attack_type"] = behavior_df.apply(classify_attack, axis=1)

    type_counts = behavior_df["attack_type"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"Brute-force": "#E74C3C", "Dictionary": "#F39C12",
              "Credential Stuffing": "#3498DB", "Hedefli": "#27AE60",
              "Düşük Aktivite": "#95A5A6"}

    # IP sayısı bazlı
    c = [colors.get(t, "#999") for t in type_counts.index]
    axes[0].pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
                colors=c, startangle=90, textprops={"fontsize": 9})
    axes[0].set_title("IP Sayısına Göre")

    # Toplam deneme bazlı
    attempt_by_type = behavior_df.groupby("attack_type")["total_attempts"].sum().reindex(type_counts.index)
    c2 = [colors.get(t, "#999") for t in attempt_by_type.index]
    axes[1].pie(attempt_by_type.values, labels=attempt_by_type.index, autopct="%1.1f%%",
                colors=c2, startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Toplam Deneme Sayısına Göre")

    plt.suptitle("Saldırı Tipi Sınıflandırması", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "D2_attack_classification.png")); plt.close(fig)
    print(f"    → D2_attack_classification.png")

    # İstatistik özeti kaydet
    summary = behavior_df.groupby("attack_type").agg(
        ip_count=("src_ip", "count"),
        total_attempts=("total_attempts", "sum"),
        avg_attempts=("total_attempts", "mean"),
        avg_unique_users=("unique_users", "mean"),
        avg_unique_passes=("unique_passes", "mean"),
    ).round(1)
    summary.to_csv(os.path.join(save_dir, "..", "data", "attack_type_summary.csv"))
    print(f"    → data/attack_type_summary.csv")

    del behavior_df; gc.collect()


def fig_D3_credential_stuffing(stuffing_df, save_dir):
    print("\n  [D3] Credential stuffing göstergeleri...")

    if stuffing_df is None or len(stuffing_df) == 0:
        print("    Veri yok — atlanıyor."); return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top-20 en çok paylaşılan kombinasyon (farklı IP sayısına göre)
    top20 = stuffing_df.head(20)
    labels = [f"{r['username']}:{r['password']}" for _, r in top20.iterrows()]
    labels = [l[:30] for l in labels]  # Uzun olanları kırp

    axes[0].barh(range(len(labels)), top20["unique_ips"].values,
                 color=C_DARK, alpha=0.85, edgecolor="white")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7, family="monospace")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Benzersiz IP Sayısı")
    axes[0].set_title("En Çok Paylaşılan Kombinasyonlar\n(Farklı IP'lerden Denenmiş)")

    # Benzersiz IP vs toplam deneme scatter
    axes[1].scatter(stuffing_df["unique_ips"], stuffing_df["total_attempts"],
                    color=C_ACCENT, alpha=0.6, s=30, edgecolors="white", linewidths=0.5)
    axes[1].set_xlabel("Benzersiz IP Sayısı")
    axes[1].set_ylabel("Toplam Deneme Sayısı")
    axes[1].set_title("Credential Stuffing: IP Dağılımı vs Deneme")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")

    plt.suptitle("Credential Stuffing Analizi", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "D3_credential_stuffing.png")); plt.close(fig)
    print(f"    → D3_credential_stuffing.png")


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 62 + "╗")
    print("║  HAFTA 4: KİMLİK BİLGİSİ ANALİZİ (v7 uyumlu)             ║")
    print("╚" + "═" * 62 + "╝")

    t_start = time.time()

    # Ön kontrol
    if not os.path.exists(COWRIE_PATH):
        raise FileNotFoundError(f"Cowrie Parquet bulunamadı: {COWRIE_PATH}")

    pf = pq.ParquetFile(COWRIE_PATH)
    cols = _get_columns(COWRIE_PATH)
    print(f"  Cowrie: {pf.metadata.num_rows:,} satır, {len(cols)} sütun")
    print(f"  Mevcut sütunlar: {', '.join(sorted(cols))}")

    os.makedirs(WEEK4_FIGS, exist_ok=True)
    os.makedirs(WEEK4_DATA, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # VERİ TOPLAMA
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("VERİ TOPLAMA")
    print("=" * 62)

    login_data = collect_login_data(COWRIE_PATH)
    entropy_df = collect_password_entropy_data(COWRIE_PATH)
    temporal_data = collect_temporal_credential_data(COWRIE_PATH)
    behavior_df = collect_attacker_behavior(COWRIE_PATH)
    stuffing_df = collect_credential_stuffing(COWRIE_PATH, top_n=50)
    country_pass = collect_country_passwords(COWRIE_PATH, top_countries=10, top_passwords=10)

    # ═══════════════════════════════════════════════════════════
    # A. KOMBİNASYON ANALİZİ
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("A. KOMBİNASYON ANALİZİ")
    print("=" * 62)

    if login_data:
        fig_A1_top_usernames(login_data, WEEK4_FIGS)
        fig_A2_top_passwords(login_data, WEEK4_FIGS)
        fig_A3_top_combos(login_data, WEEK4_FIGS)
        fig_A4_success_vs_failed(login_data, WEEK4_FIGS)
        fig_A5_overlap(login_data, WEEK4_FIGS)

        # Veri çıktıları kaydet
        pd.DataFrame(login_data["combo_counts"].most_common(100),
                      columns=["combo", "count"]).to_csv(
            os.path.join(WEEK4_DATA, "top100_combos.csv"), index=False)
        pd.DataFrame(login_data["user_counts"].most_common(50),
                      columns=["username", "count"]).to_csv(
            os.path.join(WEEK4_DATA, "top50_usernames.csv"), index=False)
        pd.DataFrame(login_data["pass_counts"].most_common(50),
                      columns=["password", "count"]).to_csv(
            os.path.join(WEEK4_DATA, "top50_passwords.csv"), index=False)
        print("    → data/top100_combos.csv, top50_usernames.csv, top50_passwords.csv")

    # ═══════════════════════════════════════════════════════════
    # B. ŞİFRE ENTROPİ ANALİZİ
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("B. ŞİFRE ENTROPİ ANALİZİ")
    print("=" * 62)

    if entropy_df is not None and len(entropy_df) > 0:
        fig_B1_entropy_histogram(entropy_df, WEEK4_FIGS)
        fig_B2_length_distribution(entropy_df, WEEK4_FIGS)
        fig_B3_complexity_classes(entropy_df, WEEK4_FIGS)
        fig_B4_entropy_vs_length(entropy_df, WEEK4_FIGS)
        fig_B5_entropy_categories(entropy_df, WEEK4_FIGS)

        # Entropi verisi kaydet
        entropy_df.to_csv(os.path.join(WEEK4_DATA, "password_entropy.csv"), index=False)
        print("    → data/password_entropy.csv")

        # Özet istatistikler
        print(f"\n  Şifre Entropi Özeti:")
        print(f"    Ortalama entropi: {entropy_df['entropy'].mean():.2f} bit")
        print(f"    Medyan entropi:   {entropy_df['entropy'].median():.2f} bit")
        print(f"    Ortalama uzunluk: {entropy_df['length'].mean():.1f}")
        print(f"    Medyan uzunluk:   {entropy_df['length'].median():.0f}")

    # ═══════════════════════════════════════════════════════════
    # C. ZAMANSAL & COĞRAFİ
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("C. ZAMANSAL & COĞRAFİ KREDENSİYEL PATERNLERİ")
    print("=" * 62)

    if temporal_data:
        fig_C1_hourly_login(temporal_data, WEEK4_FIGS)
        fig_C3_daily_unique_passwords(temporal_data, WEEK4_FIGS)
        fig_C4_ip_attempt_distribution(temporal_data, WEEK4_FIGS)

    if country_pass:
        fig_C2_country_passwords(country_pass, WEEK4_FIGS)

    # ═══════════════════════════════════════════════════════════
    # D. SALDIRGAN DAVRANIŞ PROFİLLERİ
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("D. SALDIRGAN DAVRANIŞ PROFİLLERİ")
    print("=" * 62)

    fig_D1_attacker_diversity(behavior_df, WEEK4_FIGS)
    fig_D2_attack_classification(behavior_df, WEEK4_FIGS)
    fig_D3_credential_stuffing(stuffing_df, WEEK4_FIGS)

    # ═══════════════════════════════════════════════════════════
    # ÖZET
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'=' * 62}")
    print(f"HAFTA 4 TAMAMLANDI — {elapsed:.0f}s")
    print(f"\n  Figürler ({WEEK4_FIGS}/):")
    for f in sorted(os.listdir(WEEK4_FIGS)):
        if f.endswith(".png"):
            print(f"    {f}")
    print(f"\n  Veri ({WEEK4_DATA}/):")
    for f in sorted(os.listdir(WEEK4_DATA)):
        print(f"    {f}")
    print(f"{'=' * 62}")