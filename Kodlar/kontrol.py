"""
Hafta 1 Doğrulama Scripti
===========================
Parquet çıktılarının doğru okunup işlendiğini kontrol eder.

Kontroller:
  1. Şema kontrolü — beklenen sütunlar mevcut mu?
  2. Satır sayısı — makul aralıkta mı?
  3. Tarih aralığı — dataset'in bilinen dönemine uyuyor mu?
  4. Örnek kayıtlar — ilk/son kayıtlar mantıklı mı?
  5. Değer aralıkları — portlar, IP formatı, event tipleri
  6. Null oranları — hangi sütunlarda ne kadar eksik?
  7. Duplikasyon kontrolü
"""

import os
import gc
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from collections import Counter

OUTPUT_DIR  = "/mnt/c/TEZ_PROJEEE/output"
CTU_PATH    = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")
COWRIE_PATH = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")

BATCH_SIZE = 500_000


def validate_schema(path, name, expected_cols):
    """Beklenen sütunların mevcut olduğunu kontrol eder."""
    print(f"\n{'─'*50}")
    print(f"ŞEMA KONTROLÜ: {name}")
    print(f"{'─'*50}")

    schema = pq.read_schema(path)
    actual_cols = [schema.field(i).name for i in range(len(schema))]

    print(f"  Toplam sütun: {len(actual_cols)}")
    print(f"  Sütunlar: {', '.join(sorted(actual_cols))}")

    missing = [c for c in expected_cols if c not in actual_cols]
    if missing:
        print(f"  ⚠ EKSİK SÜTUNLAR: {missing}")
    else:
        print(f"  ✓ Beklenen tüm sütunlar mevcut")

    # Veri tipleri
    print(f"\n  Veri tipleri:")
    for i in range(len(schema)):
        print(f"    {schema.field(i).name}: {schema.field(i).type}")

    return actual_cols


def validate_row_count(path, name, min_expected, max_expected):
    """Satır sayısının makul aralıkta olduğunu kontrol eder."""
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    status = "✓" if min_expected <= rows <= max_expected else "⚠"
    print(f"\n  Satır sayısı: {rows:,}  {status}")
    print(f"    Beklenen aralık: {min_expected:,} – {max_expected:,}")
    return rows


def validate_dates(path, name, expected_min_year, expected_max_year):
    """Tarih aralığının beklenen döneme uyduğunu kontrol eder."""
    print(f"\n  Tarih aralığı kontrolü:")
    cols = [pq.read_schema(path).field(i).name for i in range(len(pq.read_schema(path)))]

    if "date" not in cols:
        print(f"    ⚠ 'date' sütunu yok")
        return

    dates = pq.read_table(path, columns=["date"]).to_pandas()["date"]
    min_d, max_d = dates.min(), dates.max()
    print(f"    İlk tarih:  {min_d}")
    print(f"    Son tarih:   {max_d}")

    min_year = pd.Timestamp(min_d).year if hasattr(min_d, 'year') else int(str(min_d)[:4])
    max_year = pd.Timestamp(max_d).year if hasattr(max_d, 'year') else int(str(max_d)[:4])

    if expected_min_year <= min_year and max_year <= expected_max_year:
        print(f"    ✓ Tarihler beklenen aralıkta ({expected_min_year}–{expected_max_year})")
    else:
        print(f"    ⚠ Tarihler beklenenden farklı! Beklenen: {expected_min_year}–{expected_max_year}")

    del dates; gc.collect()


def validate_sample_records(path, name, n=5):
    """İlk ve son N kaydı gösterir — gözle kontrol için."""
    print(f"\n  Örnek kayıtlar ({name}):")
    pf = pq.ParquetFile(path)

    # İlk N kayıt
    first = next(pf.iter_batches(batch_size=n)).to_pandas()
    print(f"\n  İLK {n} KAYIT:")
    print(first.to_string(max_colwidth=40))

    # Son kayıtlar — son batch'i oku
    last_batch = None
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        last_batch = batch
    if last_batch is not None:
        last_df = last_batch.to_pandas().tail(n)
        print(f"\n  SON {n} KAYIT:")
        print(last_df.to_string(max_colwidth=40))
        del last_df

    del first, last_batch; gc.collect()


def validate_value_ranges(path, name):
    """Değer aralıklarının mantıklı olduğunu kontrol eder."""
    print(f"\n  Değer aralığı kontrolü:")
    cols = [pq.read_schema(path).field(i).name for i in range(len(pq.read_schema(path)))]

    # Port kontrolü
    for port_col in ["src_port", "dst_port"]:
        if port_col in cols:
            ports = pq.read_table(path, columns=[port_col]).to_pandas()[port_col]
            min_p, max_p = ports.min(), ports.max()
            valid = 0 <= min_p and max_p <= 65535
            status = "✓" if valid else "⚠"
            print(f"    {port_col}: {min_p} – {max_p}  {status}")
            del ports; gc.collect()

    # Saat kontrolü
    if "hour" in cols:
        hours = pq.read_table(path, columns=["hour"]).to_pandas()["hour"]
        min_h, max_h = hours.min(), hours.max()
        valid = 0 <= min_h and max_h <= 23
        status = "✓" if valid else "⚠"
        print(f"    hour: {min_h} – {max_h}  {status}")
        del hours; gc.collect()

    # IP format kontrolü (örneklem)
    if "src_ip" in cols:
        ips = pq.read_table(path, columns=["src_ip"]).to_pandas()["src_ip"].dropna()
        sample_ips = ips.head(10).tolist()
        ip_looks_valid = all("." in str(ip) or ":" in str(ip) for ip in sample_ips)
        status = "✓" if ip_looks_valid else "⚠"
        print(f"    src_ip format: {status}  (örnekler: {sample_ips[:3]})")
        del ips; gc.collect()


def validate_null_rates(path, name):
    """Her sütundaki null oranını hesaplar."""
    print(f"\n  Null oranları:")
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    schema = pf.schema_arrow
    col_names = [schema.field(i).name for i in range(len(schema))]

    null_counts = {}
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=col_names):
        df = batch.to_pandas()
        for col in df.columns:
            null_counts[col] = null_counts.get(col, 0) + int(df[col].isnull().sum())
        del df; gc.collect()

    for col in col_names:
        cnt = null_counts.get(col, 0)
        pct = cnt / total_rows * 100
        marker = ""
        if pct == 100: marker = " ← TAMAMEN BOŞ!"
        elif pct > 50: marker = " ← Dikkat"
        if cnt > 0:
            print(f"    {col}: {cnt:,} ({pct:.1f}%){marker}")

    all_null = [c for c, cnt in null_counts.items() if cnt == total_rows]
    if all_null:
        print(f"\n    ⚠ Tamamen boş sütunlar: {all_null}")
    else:
        print(f"    ✓ Tamamen boş sütun yok")


def validate_event_types(path):
    """Cowrie event tiplerinin beklenen sette olduğunu kontrol eder."""
    print(f"\n  Event tipi kontrolü:")
    cols = [pq.read_schema(path).field(i).name for i in range(len(pq.read_schema(path)))]

    if "eventid" not in cols:
        print(f"    eventid sütunu yok — atlanıyor")
        return

    expected_events = {
        "cowrie.session.connect",
        "cowrie.login.success",
        "cowrie.login.failed",
        "cowrie.command.input",
        "cowrie.session.closed",
        "cowrie.session.file_download",
    }

    event_counts = Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["eventid"]):
        event_counts.update(batch.to_pandas()["eventid"].dropna().value_counts().to_dict())
        gc.collect()

    actual_events = set(event_counts.keys())
    print(f"    Bulunan event tipleri:")
    for evt, cnt in sorted(event_counts.items(), key=lambda x: -x[1]):
        in_expected = "✓" if evt in expected_events else "⚠ BEKLENMİYOR"
        print(f"      {evt}: {cnt:,}  {in_expected}")

    missing = expected_events - actual_events
    if missing:
        print(f"    ⚠ Eksik event tipleri: {missing}")
    else:
        print(f"    ✓ Beklenen tüm event tipleri mevcut")


def validate_protocol_values(path):
    """CTU protokol değerlerinin mantıklı olduğunu kontrol eder."""
    print(f"\n  Protokol kontrolü:")
    cols = [pq.read_schema(path).field(i).name for i in range(len(pq.read_schema(path)))]

    if "proto" not in cols:
        print(f"    proto sütunu yok — atlanıyor")
        return

    valid_protos = {"tcp", "udp", "icmp"}
    proto_counts = Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["proto"]):
        proto_counts.update(batch.to_pandas()["proto"].dropna().astype(str).str.lower().value_counts().to_dict())
        gc.collect()

    for p, cnt in sorted(proto_counts.items(), key=lambda x: -x[1]):
        status = "✓" if p in valid_protos else "⚠"
        print(f"    {p}: {cnt:,}  {status}")


def check_duplicates(path, name, key_cols):
    """Anahtar sütunlar üzerinden duplikasyon kontrolü (örneklem)."""
    print(f"\n  Duplikasyon kontrolü ({', '.join(key_cols)}):")
    cols = [pq.read_schema(path).field(i).name for i in range(len(pq.read_schema(path)))]
    available_keys = [c for c in key_cols if c in cols]

    if not available_keys:
        print(f"    Anahtar sütunlar mevcut değil — atlanıyor")
        return

    # İlk 2M satırda duplikasyon kontrolü
    pf = pq.ParquetFile(path)
    sample_rows = []
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=available_keys):
        sample_rows.append(batch.to_pandas())
        if sum(len(d) for d in sample_rows) >= 2_000_000:
            break

    df = pd.concat(sample_rows, ignore_index=True)
    total = len(df)
    unique = len(df.drop_duplicates())
    dup_pct = (total - unique) / total * 100

    print(f"    İlk {total:,} satırda: {total - unique:,} duplikat ({dup_pct:.1f}%)")
    if dup_pct > 50:
        print(f"    ⚠ Yüksek duplikasyon oranı — araştırılmalı")
    else:
        print(f"    ✓ Kabul edilebilir düzeyde")

    del df, sample_rows; gc.collect()


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  HAFTA 1 DOĞRULAMA — Parquet Çıktı Kontrolü             ║")
    print("╚" + "═" * 58 + "╝")

    # ─── CTU-HORNET ───
    print("\n" + "=" * 58)
    print("CTU-HORNET-65-NINER")
    print("=" * 58)

    ctu_expected_cols = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "proto", "date", "hour", "day_of_week"
    ]

    validate_schema(CTU_PATH, "CTU-Hornet", ctu_expected_cols)
    validate_row_count(CTU_PATH, "CTU-Hornet",
                       min_expected=1_000_000, max_expected=50_000_000)
    validate_dates(CTU_PATH, "CTU-Hornet",
                   expected_min_year=2023, expected_max_year=2025)
    validate_value_ranges(CTU_PATH, "CTU-Hornet")
    validate_protocol_values(CTU_PATH)
    validate_null_rates(CTU_PATH, "CTU-Hornet")
    validate_sample_records(CTU_PATH, "CTU-Hornet", n=3)
    check_duplicates(CTU_PATH, "CTU-Hornet",
                     key_cols=["timestamp", "src_ip", "dst_ip", "src_port", "dst_port"])

    # ─── COWRIE ───
    print("\n\n" + "=" * 58)
    print("COWRIE CYBERLAB")
    print("=" * 58)

    cowrie_expected_cols = [
        "timestamp", "src_ip", "eventid", "date", "hour", "day_of_week"
    ]

    validate_schema(COWRIE_PATH, "Cowrie", cowrie_expected_cols)
    validate_row_count(COWRIE_PATH, "Cowrie",
                       min_expected=10_000_000, max_expected=200_000_000)
    validate_dates(COWRIE_PATH, "Cowrie",
                   expected_min_year=2019, expected_max_year=2021)
    validate_value_ranges(COWRIE_PATH, "Cowrie")
    validate_event_types(COWRIE_PATH)
    validate_null_rates(COWRIE_PATH, "Cowrie")
    validate_sample_records(COWRIE_PATH, "Cowrie", n=3)
    check_duplicates(COWRIE_PATH, "Cowrie",
                     key_cols=["timestamp", "src_ip", "eventid"])

    print("\n" + "=" * 58)
    print("DOĞRULAMA TAMAMLANDI")
    print("=" * 58)
    print("""
Kontrol listesi:
  ✓ = Sorunsuz
  ⚠ = İncelenmeli

Yukarıdaki çıktıları kontrol et:
  1. Sütun isimleri ve tipleri doğru mu?
  2. Satır sayıları makul mü?
  3. Tarih aralıkları dataset'in bilinen dönemine uyuyor mu?
  4. Port değerleri 0-65535 aralığında mı?
  5. IP adresleri geçerli formatta mı?
  6. Event tipleri beklenen sette mi?
  7. Null oranları kabul edilebilir mi?
  8. Örnek kayıtlar mantıklı görünüyor mu?
""")