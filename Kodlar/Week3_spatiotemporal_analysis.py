"""
Hafta 3: Mekansal-Zamansal Analiz (v7 — Bellek-Dostu)
=======================================================
Hafta 1-2 çıktılarını okur:
  - ctu_hornet_processed.parquet  (GeoLite2 zenginleştirilmiş + honeypot_city)
  - cowrie_processed.parquet      (null sütunlar temizlenmiş)

Analizler & Çıktılar:
  ═══════════════════════════════════════════════════════════
  A. MEKANSAL ANALİZ  (Folium — interaktif HTML haritaları)
  ═══════════════════════════════════════════════════════════
    A1. Küresel saldırı ısı haritası (kaynak IP → lat/lon)
        — CTU-Hornet: src_ip GeoLite2 koordinatları
        — Cowrie:     geo_latitude / geo_longitude
    A2. Honeypot lokasyon haritası (CTU dst_ip → honeypot_city)
    A3. Ülke bazlı choropleth haritası (saldırı yoğunluğu)
    A4. Cowrie oturum yoğunluk haritası (HeatMapWithTime — saatlik)

  ═══════════════════════════════════════════════════════════
  B. ZAMANSAL ANALİZ  (Matplotlib — yüksek çözünürlüklü PNG)
  ═══════════════════════════════════════════════════════════
    B1. Saat × Gün ısı matrisi (heatmap) — her iki dataset
    B2. Haftalık periyodisite analizi (Pazartesi–Pazar döngüsü)
    B3. Aylık trend karşılaştırması
    B4. İş saatleri vs gece saatleri oranı
    B5. Top-10 ülke × saat çapraz analizi (CTU)
    B6. Saldırı yoğunluğu zaman serisi (hareketli ortalama)

  ═══════════════════════════════════════════════════════════
  C. MEKANSAL-ZAMANSAL KESİŞİM
  ═══════════════════════════════════════════════════════════
    C1. Kıta bazlı saatlik profil karşılaştırması
    C2. Top-5 ülke günlük aktivite trendi

  Tüm çıktılar → OUTPUT_DIR/week3/
"""

import os
import gc
import time
import json
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
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# YAPILANDIRMA
# =====================================================================

OUTPUT_DIR    = "/mnt/c/TEZ_PROJEEE/output"
WEEK3_DIR     = os.path.join(OUTPUT_DIR, "week3")
WEEK3_MAPS    = os.path.join(WEEK3_DIR, "maps")       # Folium HTML
WEEK3_FIGS    = os.path.join(WEEK3_DIR, "figures")     # Matplotlib PNG

CTU_PATH      = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")
COWRIE_PATH   = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")
GEOLITE2_PATH = "/mnt/c/TEZ_PROJEEE/Data/GeoLite2-City/GeoLite2-City.mmdb"

BATCH_SIZE    = 500_000

# Matplotlib ayarları — tez kalitesi
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

# Renk paletleri
C_CTU    = "#2E86AB"
C_COWRIE = "#A23B72"
C_ACCENT = "#F18F01"
C_DARK   = "#1B1B2F"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_TR    = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

BUSINESS_HOURS = (8, 18)  # UTC


# =====================================================================
# YARDIMCI FONKSİYONLAR — Chunk-chunk okuma
# =====================================================================

def _col_exists(parquet_path, col_name):
    schema = pq.read_schema(parquet_path)
    return col_name in [schema.field(i).name for i in range(len(schema))]


def _get_columns(parquet_path):
    schema = pq.read_schema(parquet_path)
    return [schema.field(i).name for i in range(len(schema))]


def chunked_value_counts(parquet_path, column, top_n=None):
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[column]):
        counter.update(batch.to_pandas()[column].dropna().value_counts().to_dict())
        gc.collect()
    return dict(counter.most_common(top_n)) if top_n else dict(counter)


def chunked_two_col_counter(parquet_path, col_a, col_b):
    """(col_a_val, col_b_val) → count"""
    counter = Counter()
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[col_a, col_b]):
        df = batch.to_pandas().dropna(subset=[col_a, col_b])
        for a, b in zip(df[col_a], df[col_b]):
            counter[(a, b)] += 1
        del df; gc.collect()
    return counter


def chunked_collect_coords(parquet_path, lat_col, lon_col, sample_max=500_000):
    """Koordinat çiftlerini chunk-chunk toplar, örnekler."""
    coords = []
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[lat_col, lon_col]):
        df = batch.to_pandas().dropna(subset=[lat_col, lon_col])
        df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]
        df = df[(df[lat_col].between(-90, 90)) & (df[lon_col].between(-180, 180))]
        coords.extend(list(zip(df[lat_col].values, df[lon_col].values)))
        del df; gc.collect()
        if len(coords) >= sample_max * 3:
            break

    if len(coords) > sample_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), size=sample_max, replace=False)
        coords = [coords[i] for i in idx]

    return coords


def chunked_geo_lookup(parquet_path, ip_col, geolite2_path, sample_max=500_000):
    """IP sütunundan GeoLite2 ile koordinat çeker. Benzersiz IP'leri lookup yapar."""
    import geoip2.database

    unique_ips = set()
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[ip_col]):
        unique_ips.update(batch.to_pandas()[ip_col].dropna().unique())
        gc.collect()
        if len(unique_ips) >= 200_000:
            break

    reader = geoip2.database.Reader(geolite2_path)
    ip_coords = {}
    for ip in unique_ips:
        try:
            r = reader.city(str(ip))
            lat, lon = r.location.latitude, r.location.longitude
            if lat and lon:
                ip_coords[ip] = (lat, lon)
        except Exception:
            pass
    reader.close()
    print(f"    {len(unique_ips):,} IP → {len(ip_coords):,} koordinat")

    # IP → count eşleştirmesi
    ip_counts = Counter()
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[ip_col]):
        ip_counts.update(batch.to_pandas()[ip_col].dropna().value_counts().to_dict())
        gc.collect()

    weighted_coords = []
    for ip, (lat, lon) in ip_coords.items():
        cnt = ip_counts.get(ip, 1)
        repeat = min(cnt, 50)  # Ağırlık için tekrar, ama sınırlı
        weighted_coords.extend([(lat, lon)] * repeat)

    if len(weighted_coords) > sample_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(weighted_coords), size=sample_max, replace=False)
        weighted_coords = [weighted_coords[i] for i in idx]

    del ip_coords, ip_counts, unique_ips
    gc.collect()
    return weighted_coords


# =====================================================================
# A. MEKANSAL ANALİZ — FOLİUM HARİTALARI
# =====================================================================

def map_A1_global_heatmap(ctu_path, cowrie_path, geolite2_path, save_dir):
    """
    A1: Küresel saldırı ısı haritası — kaynak IP koordinatları.
    CTU: src_ip → GeoLite2 lookup
    Cowrie: geo_latitude / geo_longitude (veride mevcut)
    """
    import folium
    from folium.plugins import HeatMap

    print("\n  [A1] Küresel saldırı ısı haritası...")

    # --- CTU koordinatları (src_ip → GeoLite2) ---
    print("    CTU-Hornet: src_ip → GeoLite2 lookup...")
    ctu_coords = chunked_geo_lookup(ctu_path, "src_ip", geolite2_path, sample_max=300_000)

    # --- Cowrie koordinatları (mevcut geo sütunları) ---
    cowrie_cols = _get_columns(cowrie_path)
    if "geo_latitude" in cowrie_cols and "geo_longitude" in cowrie_cols:
        print("    Cowrie: geo_latitude/geo_longitude okuyor...")
        cowrie_coords = chunked_collect_coords(
            cowrie_path, "geo_latitude", "geo_longitude", sample_max=300_000
        )
    else:
        print("    Cowrie: geo koordinat sütunu yok — atlanıyor")
        cowrie_coords = []

    # --- Birleşik harita ---
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")

    if ctu_coords:
        HeatMap(
            ctu_coords, name="CTU-Hornet Saldırı Kaynakları",
            radius=8, blur=12, max_zoom=6,
            gradient={0.2: "blue", 0.5: "lime", 0.8: "yellow", 1.0: "red"}
        ).add_to(m)

    if cowrie_coords:
        fg = folium.FeatureGroup(name="Cowrie Saldırı Kaynakları")
        HeatMap(
            cowrie_coords, radius=8, blur=12, max_zoom=6,
            gradient={0.2: "purple", 0.5: "magenta", 0.8: "orange", 1.0: "red"}
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl().add_to(m)

    path = os.path.join(save_dir, "A1_global_attack_heatmap.html")
    m.save(path)
    print(f"    → {path}")
    print(f"    CTU: {len(ctu_coords):,} nokta, Cowrie: {len(cowrie_coords):,} nokta")

    del ctu_coords, cowrie_coords
    gc.collect()


def map_A2_honeypot_locations(ctu_path, geolite2_path, save_dir):
    """
    A2: Honeypot lokasyon haritası — dst_ip (honeypot) konumları.
    Her honeypot şehri bir marker ile gösterilir, boyut = akış sayısına orantılı.
    """
    import folium
    import geoip2.database

    print("\n  [A2] Honeypot lokasyon haritası...")

    # Benzersiz dst_ip ve flow sayıları
    dst_counts = Counter()
    pf = pq.ParquetFile(ctu_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["dst_ip"]):
        dst_counts.update(batch.to_pandas()["dst_ip"].dropna().value_counts().to_dict())
        gc.collect()

    # GeoLite2 ile koordinat çek
    reader = geoip2.database.Reader(geolite2_path)
    honeypot_info = {}  # city → {lat, lon, total_flows, ips: set}

    for ip, count in dst_counts.items():
        try:
            r = reader.city(str(ip))
            city = r.city.name or "Unknown"
            lat, lon = r.location.latitude, r.location.longitude
            if not lat or not lon:
                continue
            if city not in honeypot_info:
                honeypot_info[city] = {"lat": lat, "lon": lon, "flows": 0, "ips": set()}
            honeypot_info[city]["flows"] += count
            honeypot_info[city]["ips"].add(ip)
        except Exception:
            pass
    reader.close()

    print(f"    {len(honeypot_info)} benzersiz honeypot lokasyonu bulundu")

    m = folium.Map(location=[30, 0], zoom_start=2, tiles="CartoDB positron")

    for city, info in sorted(honeypot_info.items(), key=lambda x: x[1]["flows"], reverse=True):
        radius = min(max(np.log10(info["flows"] + 1) * 5, 5), 30)
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=radius,
            color="#2E86AB",
            fill=True,
            fill_color="#2E86AB",
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{city}</b><br>"
                f"Akış: {info['flows']:,}<br>"
                f"IP sayısı: {len(info['ips'])}",
                max_width=250
            ),
            tooltip=f"{city}: {info['flows']:,} akış"
        ).add_to(m)

    path = os.path.join(save_dir, "A2_honeypot_locations.html")
    m.save(path)
    print(f"    → {path}")

    del dst_counts, honeypot_info
    gc.collect()


def map_A3_choropleth(ctu_path, cowrie_path, save_dir):
    """
    A3: Ülke bazlı saldırı yoğunluk haritası.
    Her ülke merkez koordinatına orantılı boyutta CircleMarker.
    Renk: logaritmik saldırı sayısına göre (sarı→kırmızı).
    Harici GeoJSON gerektirmez — pycountry + GeoLite2 ile çalışır.
    """
    import folium
    import geoip2.database

    print("\n  [A3] Ülke bazlı saldırı yoğunluk haritası...")

    # --- Ülke sayılarını topla ---
    ctu_cols = _get_columns(ctu_path)
    ctu_cc_col = next((c for c in ["src_country_code", "geo_country_code"] if c in ctu_cols), None)

    cowrie_cols = _get_columns(cowrie_path)
    cowrie_cc_col = next((c for c in ["geo_country_code", "src_country_code"] if c in cowrie_cols), None)

    country_total = Counter()

    if ctu_cc_col:
        ctu_countries = chunked_value_counts(ctu_path, ctu_cc_col)
        country_total.update(ctu_countries)
        print(f"    CTU: {len(ctu_countries)} ülke")
        del ctu_countries

    if cowrie_cc_col:
        cowrie_countries = chunked_value_counts(cowrie_path, cowrie_cc_col)
        country_total.update(cowrie_countries)
        print(f"    Cowrie: {len(cowrie_countries)} ülke")
        del cowrie_countries

    if not country_total:
        print("    Ülke kodu sütunu bulunamadı — atlanıyor.")
        return

    # --- Ülke merkez koordinatları (GeoLite2'den IP örnekleriyle tahmin) ---
    # Her ülke koduna ait IP'lerden birinin koordinatını ülke merkezi olarak kullan
    # Daha güvenilir: birden fazla IP'nin ortalaması
    print("    Ülke merkez koordinatları hesaplanıyor...")

    country_coords = {}  # cc → (lat, lon)

    # CTU'dan ülke-koordinat eşleşmesi
    if ctu_cc_col:
        reader = geoip2.database.Reader(GEOLITE2_PATH)
        # Benzersiz IP'lerin ülke+koordinatlarını topla
        ip_sample = set()
        pf = pq.ParquetFile(ctu_path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["src_ip"]):
            ip_sample.update(batch.to_pandas()["src_ip"].dropna().unique())
            gc.collect()
            if len(ip_sample) >= 100_000:
                break

        cc_lats = defaultdict(list)
        cc_lons = defaultdict(list)
        for ip in ip_sample:
            try:
                r = reader.city(str(ip))
                cc = r.country.iso_code
                lat, lon = r.location.latitude, r.location.longitude
                if cc and lat and lon:
                    cc_lats[cc].append(lat)
                    cc_lons[cc].append(lon)
            except Exception:
                pass

        reader.close()
        for cc in cc_lats:
            country_coords[cc] = (
                np.mean(cc_lats[cc]),
                np.mean(cc_lons[cc])
            )
        del ip_sample, cc_lats, cc_lons; gc.collect()

    # Cowrie'den eksik ülkelerin koordinatlarını tamamla
    if cowrie_cc_col and ("geo_latitude" in cowrie_cols and "geo_longitude" in cowrie_cols):
        missing = [cc for cc in country_total if cc not in country_coords]
        if missing:
            cc_lats = defaultdict(list)
            cc_lons = defaultdict(list)
            pf = pq.ParquetFile(cowrie_path)
            for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                          columns=[cowrie_cc_col, "geo_latitude", "geo_longitude"]):
                df = batch.to_pandas().dropna(subset=[cowrie_cc_col, "geo_latitude", "geo_longitude"])
                df = df[df[cowrie_cc_col].isin(missing)]
                for _, row in df.iterrows():
                    cc = row[cowrie_cc_col]
                    cc_lats[cc].append(row["geo_latitude"])
                    cc_lons[cc].append(row["geo_longitude"])
                del df; gc.collect()
                # Tüm eksikler tamamlandıysa çık
                if all(cc in cc_lats for cc in missing):
                    break

            for cc in cc_lats:
                country_coords[cc] = (np.mean(cc_lats[cc]), np.mean(cc_lons[cc]))
            del cc_lats, cc_lons; gc.collect()

    # Koordinatı bulunan ülkeleri filtrele
    mapped_countries = {cc: cnt for cc, cnt in country_total.items() if cc in country_coords}
    print(f"    {len(mapped_countries)}/{len(country_total)} ülke koordinatla eşleştirildi")

    if not mapped_countries:
        print("    Koordinat eşleşmesi başarısız — atlanıyor.")
        return

    # --- Ülke isimleri (pycountry) ---
    try:
        import pycountry
        def _country_name(cc):
            c = pycountry.countries.get(alpha_2=cc)
            return c.name if c else cc
    except ImportError:
        def _country_name(cc):
            return cc

    # --- Harita oluştur ---
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    max_log = np.log10(max(mapped_countries.values()) + 1)

    # Renk skalası: sarı → turuncu → kırmızı
    import matplotlib.colors as mcolors
    cmap = plt.cm.YlOrRd

    for cc, count in sorted(mapped_countries.items(), key=lambda x: x[1]):
        lat, lon = country_coords[cc]
        log_val = np.log10(count + 1)
        normalized = log_val / max_log  # 0-1 arası

        # Boyut: log ölçekli, 3-25 px arası
        radius = max(3, min(25, log_val * 4))

        # Renk
        rgba = cmap(normalized)
        hex_color = mcolors.to_hex(rgba)

        name = _country_name(cc)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(
                f"<b>{name} ({cc})</b><br>"
                f"Toplam saldırı: {count:,}<br>"
                f"log₁₀: {log_val:.2f}",
                max_width=250
            ),
            tooltip=f"{name}: {count:,}"
        ).add_to(m)

    # Basit legend (HTML)
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:5px;
                border:1px solid #ccc; font-size:12px;">
        <b>Saldırı Yoğunluğu</b><br>
        <span style="color:#FFFFB2;">●</span> Düşük<br>
        <span style="color:#FD8D3C;">●</span> Orta<br>
        <span style="color:#BD0026;">●</span> Yüksek<br>
        <span style="font-size:10px;">(boyut ve renk log₁₀ ölçekli)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = os.path.join(save_dir, "A3_country_attack_intensity.html")
    m.save(path)
    print(f"    → {path} ({len(mapped_countries)} ülke)")

    del country_total, mapped_countries, country_coords
    gc.collect()


def map_A4_heatmap_with_time(cowrie_path, save_dir):
    """
    A4: Cowrie HeatMapWithTime — saatlik animasyonlu ısı haritası.
    Her saat için ayrı bir ısı katmanı.
    """
    import folium
    from folium.plugins import HeatMapWithTime

    print("\n  [A4] Cowrie saatlik animasyonlu ısı haritası...")

    cowrie_cols = _get_columns(cowrie_path)
    if "geo_latitude" not in cowrie_cols or "geo_longitude" not in cowrie_cols:
        print("    geo_latitude/geo_longitude yok — atlanıyor.")
        return

    # Saat → koordinat listesi
    hourly_coords = defaultdict(list)
    pf = pq.ParquetFile(cowrie_path)
    sample_per_hour = 20_000
    hour_counts = Counter()

    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                  columns=["hour", "geo_latitude", "geo_longitude"]):
        df = batch.to_pandas().dropna(subset=["geo_latitude", "geo_longitude", "hour"])
        df = df[(df["geo_latitude"] != 0) & (df["geo_longitude"] != 0)]
        df = df[(df["geo_latitude"].between(-90, 90)) & (df["geo_longitude"].between(-180, 180))]

        for _, row in df.iterrows():
            h = int(row["hour"])
            if hour_counts[h] < sample_per_hour:
                hourly_coords[h].append([float(row["geo_latitude"]), float(row["geo_longitude"])])
                hour_counts[h] += 1

        del df; gc.collect()

        if all(hour_counts[h] >= sample_per_hour for h in range(24)):
            break

    # 24 saatlik liste oluştur
    time_data = []
    time_index = []
    for h in range(24):
        coords = hourly_coords.get(h, [])
        time_data.append(coords)
        time_index.append(f"{h:02d}:00 UTC")

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    HeatMapWithTime(
        time_data,
        index=time_index,
        radius=8,
        auto_play=True,
        speed_step=0.5,
        max_opacity=0.8,
    ).add_to(m)

    path = os.path.join(save_dir, "A4_cowrie_hourly_heatmap.html")
    m.save(path)
    total_pts = sum(len(c) for c in time_data)
    print(f"    → {path} ({total_pts:,} nokta, 24 saat)")

    del hourly_coords, time_data
    gc.collect()


# =====================================================================
# B. ZAMANSAL ANALİZ — MATPLOTLİB
# =====================================================================

def fig_B1_hour_day_heatmatrix(ctu_path, cowrie_path, save_dir):
    """
    B1: Saat × Gün ısı matrisi — her iki dataset.
    X: Saat (0-23), Y: Gün (Pazartesi-Pazar), Renk: kayıt sayısı.
    """
    print("\n  [B1] Saat × Gün ısı matrisi...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (path, title, cmap) in enumerate([
        (ctu_path, "CTU-Hornet", "Blues"),
        (cowrie_path, "Cowrie", "Purples"),
    ]):
        counter = chunked_two_col_counter(path, "day_of_week", "hour")

        matrix = np.zeros((7, 24))
        for (day, hour), count in counter.items():
            if day in DAY_ORDER:
                di = DAY_ORDER.index(day)
                matrix[di, int(hour)] = count

        ax = axes[idx]
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.set_yticks(range(7))
        ax.set_yticklabels(DAY_TR, fontsize=9)
        ax.set_xlabel("Saat (UTC)")
        ax.set_ylabel("Gün")
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Kayıt Sayısı", fontsize=9)

        del counter; gc.collect()

    plt.suptitle("Saat × Gün Saldırı Yoğunluğu Matrisi", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path_out = os.path.join(save_dir, "B1_hour_day_heatmatrix.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")


def fig_B2_weekly_periodicity(ctu_path, cowrie_path, save_dir):
    """
    B2: Haftalık periyodisite — gün bazlı ortalama saldırı sayısı.
    Her gün için ortalama (toplam / hafta sayısı) hesaplanır.
    """
    print("\n  [B2] Haftalık periyodisite...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (path, title, color) in enumerate([
        (ctu_path, "CTU-Hornet", C_CTU),
        (cowrie_path, "Cowrie", C_COWRIE),
    ]):
        day_counts = chunked_value_counts(path, "day_of_week")

        # Tarih aralığından hafta sayısını hesapla
        dates = pq.read_table(path, columns=["date"]).to_pandas()["date"]
        n_weeks = max((dates.max() - dates.min()).days / 7, 1)
        del dates; gc.collect()

        vals = [day_counts.get(d, 0) / n_weeks for d in DAY_ORDER]

        ax = axes[idx]
        bars = ax.bar(DAY_TR, vals, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Ortalama Günlük Kayıt")
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

        # Değer etiketleri
        for bar, val in zip(bars, vals):
            label = f"{val/1e3:.1f}K" if val >= 1e3 else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    label, ha="center", va="bottom", fontsize=8)

        del day_counts; gc.collect()

    plt.suptitle("Haftalık Periyodisite (Ortalama Günlük Saldırı)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path_out = os.path.join(save_dir, "B2_weekly_periodicity.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")


def fig_B3_monthly_trend(ctu_path, cowrie_path, save_dir):
    """
    B3: Aylık trend — her iki dataset için ay bazlı kayıt sayısı.
    """
    print("\n  [B3] Aylık trend karşılaştırması...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for idx, (path, title, color) in enumerate([
        (ctu_path, "CTU-Hornet (Nisan–Temmuz 2024)", C_CTU),
        (cowrie_path, "Cowrie (Mayıs 2019–Şubat 2020)", C_COWRIE),
    ]):
        # Chunk-chunk ay sayımı
        month_counts = Counter()
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["date"]):
            dates = batch.to_pandas()["date"].dropna()
            for d in dates:
                month_counts[str(d)[:7]] += 1
            del dates; gc.collect()

        months = sorted(month_counts.keys())
        vals = [month_counts[m] for m in months]

        ax = axes[idx]
        bars = ax.bar(range(len(months)), vals, color=color, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Kayıt Sayısı")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

        for bar, val in zip(bars, vals):
            label = f"{val/1e6:.1f}M" if val >= 1e6 else f"{val/1e3:.0f}K" if val >= 1e3 else str(val)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    label, ha="center", va="bottom", fontsize=8)

        del month_counts; gc.collect()

    plt.suptitle("Aylık Saldırı Trendi", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path_out = os.path.join(save_dir, "B3_monthly_trend.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")


def fig_B4_business_vs_night(ctu_path, cowrie_path, save_dir):
    """
    B4: İş saatleri (08-18 UTC) vs gece saatleri oranı.
    Pasta + karşılaştırmalı çubuk grafik.
    """
    print("\n  [B4] İş saatleri vs gece saatleri...")

    results = {}

    for path, name in [(ctu_path, "CTU-Hornet"), (cowrie_path, "Cowrie")]:
        hour_counts = Counter()
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["hour"]):
            hour_counts.update(batch.to_pandas()["hour"].dropna().astype(int).value_counts().to_dict())
            gc.collect()

        business = sum(hour_counts.get(h, 0) for h in range(BUSINESS_HOURS[0], BUSINESS_HOURS[1]))
        night = sum(hour_counts.get(h, 0) for h in range(24)) - business
        results[name] = {"business": business, "night": night}
        del hour_counts; gc.collect()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors_pie = ["#F18F01", "#1B1B2F"]
    labels_pie = [f"İş Saatleri\n({BUSINESS_HOURS[0]:02d}–{BUSINESS_HOURS[1]:02d} UTC)",
                  "Gece/Dışı Saatler"]

    for idx, (name, color) in enumerate([("CTU-Hornet", C_CTU), ("Cowrie", C_COWRIE)]):
        data = results[name]
        ax = axes[idx]
        ax.pie(
            [data["business"], data["night"]],
            labels=labels_pie,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 9}
        )
        ax.set_title(name, fontsize=12, fontweight="bold")

    # Karşılaştırma çubuğu
    ax3 = axes[2]
    x = np.arange(2)
    width = 0.35
    ctu_d = results["CTU-Hornet"]
    cow_d = results["Cowrie"]
    ctu_pct = [ctu_d["business"]/(ctu_d["business"]+ctu_d["night"])*100,
               ctu_d["night"]/(ctu_d["business"]+ctu_d["night"])*100]
    cow_pct = [cow_d["business"]/(cow_d["business"]+cow_d["night"])*100,
               cow_d["night"]/(cow_d["business"]+cow_d["night"])*100]

    ax3.bar(x - width/2, ctu_pct, width, label="CTU-Hornet", color=C_CTU, alpha=0.85)
    ax3.bar(x + width/2, cow_pct, width, label="Cowrie", color=C_COWRIE, alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["İş Saatleri", "Gece/Dışı"])
    ax3.set_ylabel("Oran (%)")
    ax3.set_title("Karşılaştırma")
    ax3.legend()

    plt.suptitle("İş Saatleri vs Gece Saatleri Saldırı Dağılımı", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path_out = os.path.join(save_dir, "B4_business_vs_night.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")

    del results; gc.collect()


def fig_B5_top_country_hourly(ctu_path, save_dir):
    """
    B5: Top-10 ülke × saat çapraz analizi (CTU).
    Her ülke için saatlik saldırı profili — çizgi grafik.
    """
    print("\n  [B5] Top-10 ülke × saat çapraz analizi...")

    ctu_cols = _get_columns(ctu_path)
    cc_col = next((c for c in ["src_country_code", "geo_country_code"] if c in ctu_cols), None)

    if not cc_col:
        print("    Ülke kodu sütunu bulunamadı — atlanıyor.")
        return

    # Top 10 ülke bul
    country_counts = chunked_value_counts(ctu_path, cc_col, top_n=10)
    top_countries = list(country_counts.keys())
    del country_counts; gc.collect()

    # Ülke × saat matrisi chunk-chunk
    country_hour = defaultdict(lambda: Counter())
    pf = pq.ParquetFile(ctu_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[cc_col, "hour"]):
        df = batch.to_pandas().dropna(subset=[cc_col, "hour"])
        df = df[df[cc_col].isin(top_countries)]
        for _, row in df.iterrows():
            country_hour[row[cc_col]][int(row["hour"])] += 1
        del df; gc.collect()

    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    for i, country in enumerate(top_countries):
        vals = [country_hour[country].get(h, 0) for h in hours]
        ax.plot(hours, vals, marker="o", markersize=3, label=country,
                color=colors[i], linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Saat (UTC)")
    ax.set_ylabel("Akış Sayısı")
    ax.set_title("Top-10 Kaynak Ülke — Saatlik Saldırı Profili (CTU-Hornet)")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    plt.tight_layout()
    path_out = os.path.join(save_dir, "B5_top10_country_hourly.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")

    del country_hour; gc.collect()


def fig_B6_moving_average_timeline(ctu_path, cowrie_path, save_dir):
    """
    B6: Saldırı yoğunluğu zaman serisi — 7 günlük hareketli ortalama.
    """
    print("\n  [B6] Hareketli ortalama zaman serisi...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for idx, (path, title, color, color_light) in enumerate([
        (ctu_path, "CTU-Hornet", C_CTU, "#8BC4E0"),
        (cowrie_path, "Cowrie", C_COWRIE, "#D4A5C0"),
    ]):
        daily = Counter()
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["date"]):
            daily.update(batch.to_pandas()["date"].dropna().value_counts().to_dict())
            gc.collect()

        if not daily:
            continue

        dates = sorted(daily.keys())
        vals = [daily[d] for d in dates]

        # 7 günlük hareketli ortalama
        series = pd.Series(vals, index=pd.to_datetime(dates))
        ma7 = series.rolling(window=7, center=True).mean()

        ax = axes[idx]
        ax.bar(series.index, series.values, color=color_light, alpha=0.5, width=1.0, label="Günlük")
        ax.plot(ma7.index, ma7.values, color=color, linewidth=2, label="7-gün HO")
        ax.set_ylabel("Kayıt Sayısı")
        ax.set_title(title)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

        del daily; gc.collect()

    plt.suptitle("Saldırı Yoğunluğu Zaman Serisi (7-Gün Hareketli Ortalama)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path_out = os.path.join(save_dir, "B6_moving_average_timeline.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")


# =====================================================================
# C. MEKANSAL-ZAMANSAL KESİŞİM
# =====================================================================

def fig_C1_continent_hourly(ctu_path, cowrie_path, save_dir):
    """
    C1: Kıta bazlı saatlik profil karşılaştırması.
    Her kıta için normalize edilmiş saatlik dağılım.
    """
    print("\n  [C1] Kıta bazlı saatlik profil...")

    # Hangi datasette kıta sütunu var?
    ctu_cols = _get_columns(ctu_path)
    cowrie_cols = _get_columns(cowrie_path)

    # CTU'da geo_continent olabilir (Hafta 1 GeoLite2 zenginleştirmesinden)
    # Cowrie'de geo_continent var
    target_path = None
    continent_col = None
    ds_name = ""

    if "geo_continent" in cowrie_cols:
        target_path, continent_col, ds_name = cowrie_path, "geo_continent", "Cowrie"
    elif "geo_continent" in ctu_cols:
        target_path, continent_col, ds_name = ctu_path, "geo_continent", "CTU-Hornet"

    if not target_path:
        # GeoLite2 ile CTU src_ip'den kıta bilgisi çıkar
        print("    geo_continent sütunu bulunamadı — CTU src_ip üzerinden GeoLite2 lookup yapılıyor...")
        target_path = ctu_path
        ds_name = "CTU-Hornet"
        continent_col = None  # Aşağıda özel işlem

    if continent_col:
        # Mevcut kıta sütununu kullan
        continent_hour = defaultdict(lambda: Counter())
        pf = pq.ParquetFile(target_path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[continent_col, "hour"]):
            df = batch.to_pandas().dropna(subset=[continent_col, "hour"])
            for _, row in df.iterrows():
                continent_hour[row[continent_col]][int(row["hour"])] += 1
            del df; gc.collect()
    else:
        # GeoLite2 ile kıta lookup (CTU src_ip)
        import geoip2.database

        unique_ips = set()
        pf = pq.ParquetFile(ctu_path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["src_ip"]):
            unique_ips.update(batch.to_pandas()["src_ip"].dropna().unique())
            gc.collect()

        reader = geoip2.database.Reader(GEOLITE2_PATH)
        ip_continent = {}
        for ip in unique_ips:
            try:
                r = reader.city(str(ip))
                ip_continent[ip] = r.continent.code or "??"
            except Exception:
                pass
        reader.close()
        del unique_ips; gc.collect()

        continent_hour = defaultdict(lambda: Counter())
        pf = pq.ParquetFile(ctu_path)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["src_ip", "hour"]):
            df = batch.to_pandas().dropna(subset=["src_ip", "hour"])
            df["continent"] = df["src_ip"].map(ip_continent)
            df = df.dropna(subset=["continent"])
            for _, row in df.iterrows():
                continent_hour[row["continent"]][int(row["hour"])] += 1
            del df; gc.collect()
        del ip_continent; gc.collect()

    if not continent_hour:
        print("    Kıta verisi oluşturulamadı — atlanıyor.")
        return

    # Kıta isimleri
    continent_names = {
        "AF": "Afrika", "AS": "Asya", "EU": "Avrupa",
        "NA": "K. Amerika", "SA": "G. Amerika", "OC": "Okyanusya", "AN": "Antarktika"
    }

    # En aktif 6 kıta
    continent_totals = {c: sum(h.values()) for c, h in continent_hour.items()}
    top_continents = sorted(continent_totals, key=continent_totals.get, reverse=True)[:6]

    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, len(top_continents)))
    for i, cont in enumerate(top_continents):
        vals = [continent_hour[cont].get(h, 0) for h in hours]
        total = sum(vals) or 1
        normalized = [v / total * 100 for v in vals]
        label = continent_names.get(cont, cont)
        ax.plot(hours, normalized, marker="o", markersize=4, label=f"{label} ({cont})",
                color=colors[i], linewidth=2, alpha=0.85)

    ax.set_xlabel("Saat (UTC)")
    ax.set_ylabel("Oransal Yoğunluk (%)")
    ax.set_title(f"Kıta Bazlı Saatlik Saldırı Profili — {ds_name}")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Gece/gündüz bölgesi
    ax.axvspan(0, BUSINESS_HOURS[0], alpha=0.05, color="navy", label="_")
    ax.axvspan(BUSINESS_HOURS[1], 24, alpha=0.05, color="navy", label="_")

    plt.tight_layout()
    path_out = os.path.join(save_dir, "C1_continent_hourly_profile.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")

    del continent_hour; gc.collect()


def fig_C2_top5_country_daily(ctu_path, save_dir):
    """
    C2: Top-5 ülke günlük aktivite trendi — çizgi grafik.
    """
    print("\n  [C2] Top-5 ülke günlük aktivite trendi...")

    ctu_cols = _get_columns(ctu_path)
    cc_col = next((c for c in ["src_country_code", "geo_country_code"] if c in ctu_cols), None)

    if not cc_col:
        print("    Ülke kodu sütunu bulunamadı — atlanıyor.")
        return

    top5 = list(chunked_value_counts(ctu_path, cc_col, top_n=5).keys())
    gc.collect()

    # Ülke × tarih chunk-chunk
    country_daily = defaultdict(lambda: Counter())
    pf = pq.ParquetFile(ctu_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[cc_col, "date"]):
        df = batch.to_pandas().dropna(subset=[cc_col, "date"])
        df = df[df[cc_col].isin(top5)]
        for _, row in df.iterrows():
            country_daily[row[cc_col]][row["date"]] += 1
        del df; gc.collect()

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top5)))

    for i, country in enumerate(top5):
        daily = country_daily[country]
        dates = sorted(daily.keys())
        vals = [daily[d] for d in dates]
        series = pd.Series(vals, index=pd.to_datetime(dates))
        ma = series.rolling(window=3, center=True).mean()
        ax.plot(ma.index, ma.values, label=country, color=colors[i], linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Tarih")
    ax.set_ylabel("Günlük Akış (3-gün HO)")
    ax.set_title("Top-5 Kaynak Ülke — Günlük Saldırı Trendi (CTU-Hornet)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    plt.tight_layout()
    path_out = os.path.join(save_dir, "C2_top5_country_daily_trend.png")
    plt.savefig(path_out)
    plt.close(fig)
    print(f"    → {path_out}")

    del country_daily; gc.collect()


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 62 + "╗")
    print("║  HAFTA 3: MEKANSAL-ZAMANSAL ANALİZ (v7 uyumlu)             ║")
    print("╚" + "═" * 62 + "╝")

    t_start = time.time()

    # Ön kontrol
    for p, name in [(CTU_PATH, "CTU-Hornet"), (COWRIE_PATH, "Cowrie")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{name} Parquet bulunamadı: {p}\n"
                f"Önce Hafta 1-2 scriptleri çalıştırılmalı."
            )
        pf = pq.ParquetFile(p)
        cols = _get_columns(p)
        print(f"  {name}: {pf.metadata.num_rows:,} satır, {len(cols)} sütun")

    os.makedirs(WEEK3_MAPS, exist_ok=True)
    os.makedirs(WEEK3_FIGS, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # A. MEKANSAL ANALİZ — Folium
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("A. MEKANSAL ANALİZ — Folium Haritaları")
    print("=" * 62)

    map_A1_global_heatmap(CTU_PATH, COWRIE_PATH, GEOLITE2_PATH, WEEK3_MAPS)
    map_A2_honeypot_locations(CTU_PATH, GEOLITE2_PATH, WEEK3_MAPS)
    map_A3_choropleth(CTU_PATH, COWRIE_PATH, WEEK3_MAPS)
    map_A4_heatmap_with_time(COWRIE_PATH, WEEK3_MAPS)

    # ═══════════════════════════════════════════════════════════
    # B. ZAMANSAL ANALİZ — Matplotlib
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("B. ZAMANSAL ANALİZ — Matplotlib")
    print("=" * 62)

    fig_B1_hour_day_heatmatrix(CTU_PATH, COWRIE_PATH, WEEK3_FIGS)
    fig_B2_weekly_periodicity(CTU_PATH, COWRIE_PATH, WEEK3_FIGS)
    fig_B3_monthly_trend(CTU_PATH, COWRIE_PATH, WEEK3_FIGS)
    fig_B4_business_vs_night(CTU_PATH, COWRIE_PATH, WEEK3_FIGS)
    fig_B5_top_country_hourly(CTU_PATH, WEEK3_FIGS)
    fig_B6_moving_average_timeline(CTU_PATH, COWRIE_PATH, WEEK3_FIGS)

    # ═══════════════════════════════════════════════════════════
    # C. MEKANSAL-ZAMANSAL KESİŞİM
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("C. MEKANSAL-ZAMANSAL KESİŞİM")
    print("=" * 62)

    fig_C1_continent_hourly(CTU_PATH, COWRIE_PATH, WEEK3_FIGS)
    fig_C2_top5_country_daily(CTU_PATH, WEEK3_FIGS)

    # ═══════════════════════════════════════════════════════════
    # ÖZET
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'=' * 62}")
    print(f"HAFTA 3 TAMAMLANDI — {elapsed:.0f}s")
    print(f"\n  Haritalar ({WEEK3_MAPS}/):")
    for f in sorted(os.listdir(WEEK3_MAPS)):
        size_mb = os.path.getsize(os.path.join(WEEK3_MAPS, f)) / 1e6
        print(f"    {f}  ({size_mb:.1f} MB)")
    print(f"\n  Figürler ({WEEK3_FIGS}/):")
    for f in sorted(os.listdir(WEEK3_FIGS)):
        print(f"    {f}")
    print(f"{'=' * 62}")