"""
Hafta 2: Temizleme & Görselleştirme (v7 — Bellek-Dostu)
=========================================================
Hafta 1 (v7) çıktılarını okur:
  - ctu_hornet_processed.parquet
  - cowrie_processed.parquet

İşlemler:
  1. Cowrie: Tamamen null olan sütunları tespit edip temizlenmiş parquet yazar
  2. CTU-Hornet: Dosya yolundan honeypot_city çıkarımı (arşiv adındaki şehir kodu)
  3. Görselleştirmeler (chunk-chunk, 16 GB RAM uyumlu):
     - Temporal: Saatlik & günlük dağılım
     - Port analizi: Top-20 hedef port
     - Protokol dağılımı
     - Ülke bazlı saldırı haritası (coğrafi)
     - Cowrie credential analizi (top kullanıcı/şifre)
     - Dataset karşılaştırma özeti
  4. Tüm figürler → OUTPUT_DIR/figures/
"""

import os
import gc
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# YAPILANDIRMA  (Hafta 1 v7 ile uyumlu)
# =====================================================================

OUTPUT_DIR  = "/mnt/c/TEZ_PROJEEE/output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

CTU_PATH    = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")
COWRIE_PATH = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")

BATCH_SIZE  = 500_000

# Hafta 1 GeoLite2 veritabanı — honeypot_city çıkarımı için (dst_ip → şehir)
GEOLITE2_PATH = "/mnt/c/TEZ_PROJEEE/Data/GeoLite2-City/GeoLite2-City.mmdb"

# Matplotlib genel ayarlar
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# Renk paletleri
PALETTE_CTU    = "#2E86AB"
PALETTE_COWRIE = "#A23B72"
PALETTE_ACCENT = "#F18F01"
PALETTE_SEQ    = plt.cm.viridis


# =====================================================================
# 1. COWRIE — NULL SÜTUN TEMİZLİĞİ
# =====================================================================

def detect_null_columns(parquet_path):
    """Chunk-chunk okuyarak tamamen null olan sütunları tespit eder."""
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    col_names = [schema.field(i).name for i in range(len(schema))]

    # Her sütun için en az bir non-null değer var mı?
    has_data = {c: False for c in col_names}

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        for col in col_names:
            if has_data[col]:
                continue  # Zaten non-null bulundu
            if col in df.columns and df[col].notna().any():
                has_data[col] = True
        del df
        gc.collect()

        # Hepsi True olduysa erken çık
        if all(has_data.values()):
            break

    null_cols = [c for c, v in has_data.items() if not v]
    return null_cols


def clean_cowrie_null_columns(parquet_path):
    """
    Cowrie parquet dosyasından tamamen null olan sütunları kaldırır.
    Temizlenmiş dosyayı aynı yola yazar (yerinde güncelleme).
    """
    print("\n" + "=" * 60)
    print("COWRIE — NULL SÜTUN TEMİZLİĞİ")
    print("=" * 60)

    null_cols = detect_null_columns(parquet_path)

    if not null_cols:
        print("  Tamamen null sütun bulunamadı — temizlik gerekmiyor.")
        return parquet_path

    print(f"  {len(null_cols)} tamamen null sütun tespit edildi:")
    for c in sorted(null_cols):
        print(f"    - {c}")

    # Hangi sütunlar kalacak?
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    all_cols = [schema.field(i).name for i in range(len(schema))]
    keep_cols = [c for c in all_cols if c not in null_cols]

    print(f"  {len(all_cols)} → {len(keep_cols)} sütun (temizlenmiş)")

    # Chunk-chunk yeni dosyaya yaz
    cleaned_path = parquet_path + ".cleaned"
    writer = None
    total = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=keep_cols):
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(cleaned_path, table.schema, compression="snappy")
        writer.write_table(table)
        total += len(batch)
        del table
        gc.collect()

    if writer:
        writer.close()

    # Eski dosyayı değiştir
    os.replace(cleaned_path, parquet_path)
    print(f"  Temizlik tamamlandı: {total:,} satır, {len(keep_cols)} sütun")
    return parquet_path


# =====================================================================
# 2. CTU-HORNET — HONEYPOT_CITY ÇIKARIMI  (dst_ip → GeoLite2)
# =====================================================================

def _build_ip_city_map(parquet_path, geolite2_path):
    """
    Parquet'teki benzersiz dst_ip'leri toplar, GeoLite2 ile şehir eşleştirir.
    Döndürür: {ip_str: city_name}
    Tüm satırları RAM'e almaz — sadece benzersiz IP seti + lookup tablosu tutar.
    """
    import geoip2.database

    # 1) Benzersiz dst_ip'leri chunk-chunk topla
    unique_ips = set()
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["dst_ip"]):
        unique_ips.update(batch.to_pandas()["dst_ip"].dropna().unique())
        gc.collect()

    print(f"  Benzersiz dst_ip: {len(unique_ips):,}")

    # 2) GeoLite2 lookup
    ip_city = {}
    reader = geoip2.database.Reader(geolite2_path)

    resolved, failed = 0, 0
    for ip in unique_ips:
        try:
            resp = reader.city(str(ip))
            city = resp.city.name or "Unknown"
            ip_city[ip] = city
            resolved += 1
        except Exception:
            ip_city[ip] = "Unknown"
            failed += 1

    reader.close()
    print(f"  GeoLite2 sonuç: {resolved:,} çözümlendi, {failed:,} başarısız")
    return ip_city


def enrich_ctu_honeypot_city(parquet_path, geolite2_path):
    """
    CTU Parquet'e honeypot_city sütunu ekler.
    dst_ip (honeypot IP) üzerinden GeoLite2 ile şehir bilgisi çıkarır.
    Chunk-chunk okur/yazar — bellek dostu.
    """
    print("\n" + "=" * 60)
    print("CTU-HORNET — HONEYPOT CITY ÇIKARIMI (dst_ip → GeoLite2)")
    print("=" * 60)

    schema = pq.read_schema(parquet_path)
    col_names = [schema.field(i).name for i in range(len(schema))]

    if "honeypot_city" in col_names:
        print("  honeypot_city zaten mevcut — atlanıyor.")
        return parquet_path

    if "dst_ip" not in col_names:
        print("  UYARI: dst_ip sütunu bulunamadı — honeypot_city eklenemiyor.")
        return parquet_path

    if not os.path.exists(geolite2_path):
        print(f"  UYARI: GeoLite2 DB bulunamadı: {geolite2_path}")
        print(f"  honeypot_city adımı atlanıyor.")
        return parquet_path

    # IP → şehir eşleme tablosu oluştur
    ip_city = _build_ip_city_map(parquet_path, geolite2_path)

    # Chunk-chunk oku, honeypot_city sütunu ekle, yeni dosyaya yaz
    enriched_path = parquet_path + ".city"
    pf = pq.ParquetFile(parquet_path)
    writer = None
    total = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        df["honeypot_city"] = df["dst_ip"].map(ip_city).fillna("Unknown")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(enriched_path, table.schema, compression="snappy")
        writer.write_table(table)
        total += len(df)
        del df, table
        gc.collect()

    if writer:
        writer.close()

    del ip_city
    gc.collect()

    os.replace(enriched_path, parquet_path)
    print(f"  honeypot_city eklendi: {total:,} kayıt")
    return parquet_path


# =====================================================================
# 3. GÖRSELLEŞTİRMELER — Chunk-chunk okuma
# =====================================================================

# ----- Yardımcı: chunk-chunk value_counts -----

def chunked_value_counts(parquet_path, column, top_n=None, filters=None):
    """Parquet'ten chunk-chunk okuyarak value_counts hesaplar."""
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[column]):
        series = batch.to_pandas()[column].dropna()
        counter.update(series.value_counts().to_dict())
        del series
        gc.collect()

    if top_n:
        return dict(counter.most_common(top_n))
    return dict(counter)


def chunked_value_counts_filtered(parquet_path, count_col, filter_col, filter_val, top_n=None):
    """Belirli bir filtreyle chunk-chunk value_counts."""
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)
    cols = list(set([count_col, filter_col]))

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        df = batch.to_pandas()
        mask = df[filter_col] == filter_val
        series = df.loc[mask, count_col].dropna()
        counter.update(series.value_counts().to_dict())
        del df, series
        gc.collect()

    if top_n:
        return dict(counter.most_common(top_n))
    return dict(counter)


def chunked_hour_distribution(parquet_path):
    """Saatlik dağılım: {hour: count}"""
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["hour"]):
        series = batch.to_pandas()["hour"].dropna().astype(int)
        counter.update(series.value_counts().to_dict())
        del series
        gc.collect()

    return dict(counter)


def chunked_daily_counts(parquet_path):
    """Günlük kayıt sayısı: {date: count}"""
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["date"]):
        series = batch.to_pandas()["date"].dropna()
        counter.update(series.value_counts().to_dict())
        del series
        gc.collect()

    return dict(sorted(counter.items()))


def chunked_two_col_counts(parquet_path, col1, col2, filter_col=None, filter_val=None, top_n=20):
    """İki sütunun kombinasyonları için chunk-chunk sayım."""
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)
    cols = list(set([col1, col2] + ([filter_col] if filter_col else [])))

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        df = batch.to_pandas()
        if filter_col and filter_val:
            df = df[df[filter_col] == filter_val]
        pairs = df[[col1, col2]].dropna()
        for _, row in pairs.iterrows():
            counter[(row[col1], row[col2])] += 1
        del df, pairs
        gc.collect()

    return dict(counter.most_common(top_n))


# =====================================================================
# GRAFİK FONKSİYONLARI
# =====================================================================

def fig_01_hourly_distribution(ctu_path, cowrie_path, save_dir):
    """Saatlik saldırı dağılımı — her iki dataset yan yana."""
    print("  [1/8] Saatlik dağılım...")

    ctu_hours = chunked_hour_distribution(ctu_path)
    cowrie_hours = chunked_hour_distribution(cowrie_path)

    hours = list(range(24))
    ctu_vals = [ctu_hours.get(h, 0) for h in hours]
    cowrie_vals = [cowrie_hours.get(h, 0) for h in hours]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(hours, ctu_vals, color=PALETTE_CTU, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_title("CTU-Hornet — Saatlik Dağılım (UTC)")
    ax1.set_xlabel("Saat")
    ax1.set_ylabel("Akış Sayısı")
    ax1.set_xticks(range(0, 24, 2))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    ax2.bar(hours, cowrie_vals, color=PALETTE_COWRIE, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_title("Cowrie — Saatlik Dağılım (UTC)")
    ax2.set_xlabel("Saat")
    ax2.set_ylabel("Olay Sayısı")
    ax2.set_xticks(range(0, 24, 2))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    plt.tight_layout()
    path = os.path.join(save_dir, "01_hourly_distribution.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")
    del ctu_hours, cowrie_hours
    gc.collect()


def fig_02_daily_timeline(ctu_path, cowrie_path, save_dir):
    """Günlük zaman serisi — her iki dataset."""
    print("  [2/8] Günlük zaman serisi...")

    ctu_daily = chunked_daily_counts(ctu_path)
    cowrie_daily = chunked_daily_counts(cowrie_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    if ctu_daily:
        dates_c = sorted(ctu_daily.keys())
        vals_c = [ctu_daily[d] for d in dates_c]
        ax1.fill_between(range(len(dates_c)), vals_c, alpha=0.3, color=PALETTE_CTU)
        ax1.plot(range(len(dates_c)), vals_c, color=PALETTE_CTU, linewidth=1)
        ax1.set_title("CTU-Hornet — Günlük Akış Sayısı")
        ax1.set_ylabel("Akış")
        # X ekseni: her 30 günde bir etiket
        step = max(1, len(dates_c) // 12)
        ax1.set_xticks(range(0, len(dates_c), step))
        ax1.set_xticklabels([str(dates_c[i]) for i in range(0, len(dates_c), step)],
                            rotation=45, ha="right", fontsize=8)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    if cowrie_daily:
        dates_w = sorted(cowrie_daily.keys())
        vals_w = [cowrie_daily[d] for d in dates_w]
        ax2.fill_between(range(len(dates_w)), vals_w, alpha=0.3, color=PALETTE_COWRIE)
        ax2.plot(range(len(dates_w)), vals_w, color=PALETTE_COWRIE, linewidth=1)
        ax2.set_title("Cowrie — Günlük Olay Sayısı")
        ax2.set_ylabel("Olay")
        step = max(1, len(dates_w) // 12)
        ax2.set_xticks(range(0, len(dates_w), step))
        ax2.set_xticklabels([str(dates_w[i]) for i in range(0, len(dates_w), step)],
                            rotation=45, ha="right", fontsize=8)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    plt.tight_layout()
    path = os.path.join(save_dir, "02_daily_timeline.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")
    del ctu_daily, cowrie_daily
    gc.collect()


def fig_03_top_ports(ctu_path, save_dir):
    """CTU-Hornet: Top 20 hedef port."""
    print("  [3/8] Top hedef portlar...")

    port_counts = chunked_value_counts(ctu_path, "dst_port", top_n=20)

    ports = list(port_counts.keys())
    counts = list(port_counts.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(ports)), counts, color=PALETTE_CTU, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(ports)))
    ax.set_yticklabels([str(p) for p in ports])
    ax.invert_yaxis()
    ax.set_xlabel("Akış Sayısı")
    ax.set_title("CTU-Hornet — En Çok Hedeflenen 20 Port")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    # Değer etiketleri
    for bar, val in zip(bars, counts):
        label = f"{val/1e6:.1f}M" if val >= 1e6 else f"{val/1e3:.0f}K" if val >= 1e3 else str(val)
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8, color="#333")

    plt.tight_layout()
    path = os.path.join(save_dir, "03_top_ports_ctu.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")
    del port_counts
    gc.collect()


def fig_04_protocol_distribution(ctu_path, save_dir):
    """CTU-Hornet: Protokol dağılımı (TCP/UDP/ICMP...)."""
    print("  [4/8] Protokol dağılımı...")

    schema = pq.read_schema(ctu_path)
    col_names = [schema.field(i).name for i in range(len(schema))]

    if "proto" not in col_names:
        print("    proto sütunu bulunamadı — atlanıyor.")
        return

    proto_counts = chunked_value_counts(ctu_path, "proto")

    labels = list(proto_counts.keys())
    sizes = list(proto_counts.values())
    total = sum(sizes)

    # Çok küçükleri "Diğer" olarak grupla
    threshold = total * 0.01
    main_labels, main_sizes, other = [], [], 0
    for l, s in zip(labels, sizes):
        if s >= threshold:
            main_labels.append(l)
            main_sizes.append(s)
        else:
            other += s
    if other > 0:
        main_labels.append("Diğer")
        main_sizes.append(other)

    colors = plt.cm.Set2(np.linspace(0, 1, len(main_labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pasta grafik
    wedges, texts, autotexts = ax1.pie(
        main_sizes, labels=main_labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.85
    )
    ax1.set_title("Protokol Dağılımı (Pasta)")

    # Çubuk grafik
    ax2.bar(main_labels, main_sizes, color=colors, edgecolor="white")
    ax2.set_title("Protokol Dağılımı (Çubuk)")
    ax2.set_ylabel("Akış Sayısı")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path = os.path.join(save_dir, "04_protocol_distribution.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")
    del proto_counts
    gc.collect()


def fig_05_country_top20(ctu_path, cowrie_path, save_dir):
    """Her iki datasetten top 20 kaynak ülke."""
    print("  [5/8] Ülke bazlı saldırı dağılımı...")

    # CTU tarafı
    ctu_schema = pq.read_schema(ctu_path)
    ctu_cols = [ctu_schema.field(i).name for i in range(len(ctu_schema))]
    ctu_country_col = None
    for candidate in ["src_country_code", "geo_country_code"]:
        if candidate in ctu_cols:
            ctu_country_col = candidate
            break

    # Cowrie tarafı
    cowrie_schema = pq.read_schema(cowrie_path)
    cowrie_cols = [cowrie_schema.field(i).name for i in range(len(cowrie_schema))]
    cowrie_country_col = None
    for candidate in ["geo_country_code", "src_country_code"]:
        if candidate in cowrie_cols:
            cowrie_country_col = candidate
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (path, col, title, color) in enumerate([
        (ctu_path, ctu_country_col, "CTU-Hornet", PALETTE_CTU),
        (cowrie_path, cowrie_country_col, "Cowrie", PALETTE_COWRIE),
    ]):
        ax = axes[idx]
        if col is None:
            ax.text(0.5, 0.5, "Ülke verisi mevcut değil", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title(f"{title} — Top 20 Ülke")
            continue

        counts = chunked_value_counts(path, col, top_n=20)
        countries = list(counts.keys())
        values = list(counts.values())

        ax.barh(range(len(countries)), values, color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries)
        ax.invert_yaxis()
        ax.set_xlabel("Kayıt Sayısı")
        ax.set_title(f"{title} — Top 20 Kaynak Ülke")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

        del counts
        gc.collect()

    plt.tight_layout()
    path_out = os.path.join(save_dir, "05_country_top20.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")


def fig_06_cowrie_credentials(cowrie_path, save_dir):
    """Cowrie: En çok denenen kullanıcı adı ve şifre."""
    print("  [6/8] Cowrie credential analizi...")

    schema = pq.read_schema(cowrie_path)
    col_names = [schema.field(i).name for i in range(len(schema))]

    has_username = "username" in col_names
    has_password = "password" in col_names

    if not has_username and not has_password:
        print("    username/password sütunu bulunamadı — atlanıyor.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Top 20 kullanıcı adı
    if has_username:
        user_counts = chunked_value_counts(cowrie_path, "username", top_n=20)
        users = list(user_counts.keys())
        u_vals = list(user_counts.values())

        axes[0].barh(range(len(users)), u_vals, color=PALETTE_COWRIE, alpha=0.85, edgecolor="white")
        axes[0].set_yticks(range(len(users)))
        axes[0].set_yticklabels(users, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Deneme Sayısı")
        axes[0].set_title("Top 20 Kullanıcı Adı")
        axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))
        del user_counts
        gc.collect()
    else:
        axes[0].text(0.5, 0.5, "username sütunu yok", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
        axes[0].set_title("Top 20 Kullanıcı Adı")

    # Top 20 şifre
    if has_password:
        pass_counts = chunked_value_counts(cowrie_path, "password", top_n=20)
        passwords = list(pass_counts.keys())
        p_vals = list(pass_counts.values())

        axes[1].barh(range(len(passwords)), p_vals, color=PALETTE_ACCENT, alpha=0.85, edgecolor="white")
        axes[1].set_yticks(range(len(passwords)))
        axes[1].set_yticklabels(passwords, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Deneme Sayısı")
        axes[1].set_title("Top 20 Şifre")
        axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))
        del pass_counts
        gc.collect()
    else:
        axes[1].text(0.5, 0.5, "password sütunu yok", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
        axes[1].set_title("Top 20 Şifre")

    plt.tight_layout()
    path = os.path.join(save_dir, "06_cowrie_credentials.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


def fig_07_cowrie_event_types(cowrie_path, save_dir):
    """Cowrie: Olay tipi dağılımı."""
    print("  [7/8] Cowrie olay tipi dağılımı...")

    schema = pq.read_schema(cowrie_path)
    col_names = [schema.field(i).name for i in range(len(schema))]

    if "eventid" not in col_names:
        print("    eventid sütunu bulunamadı — atlanıyor.")
        return

    event_counts = chunked_value_counts(cowrie_path, "eventid")

    # Sırala (büyükten küçüğe)
    sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [e[0].replace("cowrie.", "") for e in sorted_events]
    values = [e[1] for e in sorted_events]

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Olay Sayısı")
    ax.set_title("Cowrie — Olay Tipi Dağılımı")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    for bar, val in zip(bars, values):
        label = f"{val/1e6:.1f}M" if val >= 1e6 else f"{val/1e3:.0f}K" if val >= 1e3 else str(val)
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8, color="#333")

    plt.tight_layout()
    path = os.path.join(save_dir, "07_cowrie_event_types.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")
    del event_counts
    gc.collect()


def fig_08_dataset_comparison(ctu_path, cowrie_path, save_dir):
    """Özet karşılaştırma tablosu — dataset boyutları, zaman aralığı, IP sayıları."""
    print("  [8/8] Dataset karşılaştırma özeti...")

    # CTU istatistikleri
    ctu_pf = pq.ParquetFile(ctu_path)
    ctu_rows = ctu_pf.metadata.num_rows
    ctu_schema = ctu_pf.schema_arrow
    ctu_cols = len(ctu_schema)
    ctu_col_names = [ctu_schema.field(i).name for i in range(len(ctu_schema))]

    ctu_ips = pq.read_table(ctu_path, columns=["src_ip"]).to_pandas()["src_ip"].nunique()
    gc.collect()

    ctu_dates = pq.read_table(ctu_path, columns=["date"]).to_pandas()["date"]
    ctu_start, ctu_end = str(ctu_dates.min()), str(ctu_dates.max())
    del ctu_dates; gc.collect()

    # Cowrie istatistikleri
    cowrie_pf = pq.ParquetFile(cowrie_path)
    cowrie_rows = cowrie_pf.metadata.num_rows
    cowrie_schema = cowrie_pf.schema_arrow
    cowrie_cols = len(cowrie_schema)

    cowrie_ips = set()
    for batch in cowrie_pf.iter_batches(batch_size=1_000_000, columns=["src_ip"]):
        cowrie_ips.update(batch.to_pandas()["src_ip"].unique())
        gc.collect()
    cowrie_ip_count = len(cowrie_ips)
    del cowrie_ips; gc.collect()

    cowrie_dates = pq.read_table(cowrie_path, columns=["date"]).to_pandas()["date"]
    cowrie_start, cowrie_end = str(cowrie_dates.min()), str(cowrie_dates.max())
    del cowrie_dates; gc.collect()

    # Tablo figürü
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table_data = [
        ["Metrik", "CTU-Hornet-65-Niner", "Cowrie CyberLab"],
        ["Toplam Kayıt", f"{ctu_rows:,}", f"{cowrie_rows:,}"],
        ["Sütun Sayısı", str(ctu_cols), str(cowrie_cols)],
        ["Benzersiz IP", f"{ctu_ips:,}", f"{cowrie_ip_count:,}"],
        ["Zaman Aralığı", f"{ctu_start} → {ctu_end}", f"{cowrie_start} → {cowrie_end}"],
        ["Veri Tipi", "Ağ akışı (Zeek conn)", "SSH/Telnet oturum"],
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.375, 0.375],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Başlık satırı renklendirme
    for j in range(3):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Satır renklendirme
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[i, j].set_facecolor("#ecf0f1")
            else:
                table[i, j].set_facecolor("#ffffff")

    ax.set_title("Dataset Karşılaştırma Özeti", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(save_dir, "08_dataset_comparison.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  HAFTA 2: TEMİZLEME & GÖRSELLEŞTİRME (v7 uyumlu)       ║")
    print("╚" + "═" * 58 + "╝")

    t_start = time.time()

    # Ön kontrol
    for p, name in [(CTU_PATH, "CTU-Hornet"), (COWRIE_PATH, "Cowrie")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{name} Parquet bulunamadı: {p}\n"
                f"Önce Hafta 1 (week1_data_integration_v7.py) çalıştırılmalı."
            )
        pf = pq.ParquetFile(p)
        print(f"  {name}: {pf.metadata.num_rows:,} satır, {len(pf.schema_arrow)} sütun")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Cowrie null sütun temizliği ──
    COWRIE_PATH_CLEAN = clean_cowrie_null_columns(COWRIE_PATH)

    # ── 2. CTU honeypot_city çıkarımı ──
    CTU_PATH_ENRICHED = enrich_ctu_honeypot_city(CTU_PATH, GEOLITE2_PATH)

    # ── 3. Görselleştirmeler ──
    print("\n" + "=" * 60)
    print("GÖRSELLEŞTİRMELER")
    print("=" * 60)

    fig_01_hourly_distribution(CTU_PATH_ENRICHED, COWRIE_PATH_CLEAN, FIGURES_DIR)
    fig_02_daily_timeline(CTU_PATH_ENRICHED, COWRIE_PATH_CLEAN, FIGURES_DIR)
    fig_03_top_ports(CTU_PATH_ENRICHED, FIGURES_DIR)
    fig_04_protocol_distribution(CTU_PATH_ENRICHED, FIGURES_DIR)
    fig_05_country_top20(CTU_PATH_ENRICHED, COWRIE_PATH_CLEAN, FIGURES_DIR)
    fig_06_cowrie_credentials(COWRIE_PATH_CLEAN, FIGURES_DIR)
    fig_07_cowrie_event_types(COWRIE_PATH_CLEAN, FIGURES_DIR)
    fig_08_dataset_comparison(CTU_PATH_ENRICHED, COWRIE_PATH_CLEAN, FIGURES_DIR)

    # ── Özet ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"HAFTA 2 TAMAMLANDI — {elapsed:.0f}s")
    print(f"  Figürler: {FIGURES_DIR}/")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith(".png"):
            print(f"    {f}")
    print(f"{'=' * 60}")