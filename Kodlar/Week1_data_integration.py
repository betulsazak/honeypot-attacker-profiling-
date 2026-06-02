"""
Hafta 1: Veri Entegrasyonu (v7 — Bellek-Dostu)
================================================
- v6 ile aynı okuma / chunk / birleştirme mantığı
- load_* fonksiyonları DataFrame yerine dosya yolu döndürür
- Özet istatistikler sadece gerekli sütunlar okunarak hesaplanır
- create_unified_format predicate pushdown ile çalışır
- Hiçbir noktada tüm veri tek seferde RAM'e alınmaz
"""

import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import gc
import gzip
import zipfile
import tarfile
import io
import time
import shutil
from decimal import Decimal

# =====================================================================
# YAPILANDIRMA
# =====================================================================

CTU_HORNET_DIR = "/mnt/c/TEZ_PROJEEE/Data/CTU-Hornet-65-Niner/zeek"
COWRIE_DIR     = "/mnt/c/TEZ_PROJEEE/Data/Cowrie"
GEOLITE2_PATH  = "/mnt/c/TEZ_PROJEEE/Data/GeoLite2-City/GeoLite2-City.mmdb"
OUTPUT_DIR     = "/mnt/c/TEZ_PROJEEE/output"

CHUNK_SIZE = 500_000

ZEEK_LOG_PREFIXES = ("conn.",)

COWRIE_TARGET_EVENTS = {
    "cowrie.session.connect",
    "cowrie.login.success",
    "cowrie.login.failed",
    "cowrie.command.input",
    "cowrie.session.closed",
    "cowrie.session.file_download",
}

PARQUET_COMPRESSION = "snappy"


# =====================================================================
# Dosya Filtresi & Bulucu
# =====================================================================

def _is_target_log(filename):
    b = os.path.basename(filename).lower()
    if b.endswith(".json") or b.endswith(".json.gz"): return True
    if b.endswith(".log") or b.endswith(".log.gz"):
        return any(b.startswith(p) for p in ZEEK_LOG_PREFIXES)
    return False

def _find_data_files(data_dir):
    archives = []
    try:
        for e in os.scandir(data_dir):
            if e.is_file() and e.name.lower().endswith((".tar.gz",".tgz",".tar.bz2",".zip")):
                archives.append(e.path)
    except OSError as err:
        print(f"  HATA: {data_dir} ({err})")
        return []
    if archives:
        print(f"  Arşiv: {len(archives)}")
        return sorted(archives)
    found, skip = [], 0
    for dp, _, fns in os.walk(data_dir):
        for fn in fns:
            if _is_target_log(fn): found.append(os.path.join(dp, fn))
            else: skip += 1
    print(f"  Tarama: {len(found)} hedef, {skip} atlandı")
    return sorted(found)


# =====================================================================
# CTU-Hornet Dosya Okuyucular (NDJSON)
# =====================================================================

def _open_smart(filepath):
    e = filepath.lower()
    if e.endswith(".tar.gz") or e.endswith(".tgz"): yield from _iter_tar_gz(filepath)
    elif e.endswith(".tar.bz2"): yield from _iter_tar_bz2(filepath)
    elif e.endswith(".gz"): yield from _iter_gzip(filepath)
    else: yield from _iter_plain(filepath)

def _iter_plain(fp):
    fn = os.path.basename(fp)
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        for line in f: yield fn, line

def _iter_gzip(fp):
    if not _is_target_log(fp): return
    fn = os.path.basename(fp)
    with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as f:
        for line in f: yield fn, line

def _iter_tar_gz(fp):
    with tarfile.open(fp, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or not _is_target_log(m.name): continue
            f = tar.extractfile(m)
            if not f: continue
            if m.name.lower().endswith(".gz"):
                with gzip.open(f, "rt", encoding="utf-8", errors="replace") as gf:
                    for line in gf: yield m.name, line
            else:
                w = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                for line in w: yield m.name, line
                w.detach()

def _iter_tar_bz2(fp):
    with tarfile.open(fp, "r:bz2") as tar:
        for m in tar:
            if not m.isfile() or not _is_target_log(m.name): continue
            f = tar.extractfile(m)
            if not f: continue
            if m.name.lower().endswith(".gz"):
                with gzip.open(f, "rt", encoding="utf-8", errors="replace") as gf:
                    for line in gf: yield m.name, line
            else:
                w = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                for line in w: yield m.name, line
                w.detach()


# =====================================================================
# Cowrie Dosya Okuyucu (JSON Array — ijson streaming)
# =====================================================================

def _decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict): return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_decimal_to_float(i) for i in obj]
    return obj

def _iter_cowrie_events(filepath):
    import ijson
    if filepath.lower().endswith(".gz"):
        fh = gzip.open(filepath, "rb")
    else:
        fh = open(filepath, "rb")
    try:
        for session_obj in ijson.items(fh, "item"):
            if not isinstance(session_obj, dict): continue
            for sid, events in session_obj.items():
                if not isinstance(events, list): continue
                for event in events:
                    if isinstance(event, dict):
                        yield _decimal_to_float(event)
    except Exception as e:
        print(f"  Uyarı: {os.path.basename(filepath)}: {e}")
    finally:
        fh.close()


# =====================================================================
# Chunk İşleme
# =====================================================================

def _process_ctu_chunk(records):
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    rn = {"id.orig_h":"src_ip","id.orig_p":"src_port","id.resp_h":"dst_ip","id.resp_p":"dst_port"}
    df = df.rename(columns={k:v for k,v in rn.items() if k in df.columns})
    for c in ["proto","service","conn_state","day_of_week"]:
        if c in df.columns: df[c] = df[c].astype("category")
    for c in ["src_port","dst_port","orig_bytes","resp_bytes","missed_bytes",
              "orig_pkts","resp_pkts","orig_ip_bytes","resp_ip_bytes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    return df

def _process_cowrie_record(record):
    geo = record.pop("geolocation_data", None)
    if geo and isinstance(geo, dict):
        record["geo_country_code"] = geo.get("country_code3", "")
        record["geo_country_name"] = geo.get("country_name", "")
        record["geo_continent"] = geo.get("continent_code", "")
        loc = geo.get("location", {})
        if isinstance(loc, dict):
            record["geo_latitude"] = loc.get("lat", geo.get("latitude"))
            record["geo_longitude"] = loc.get("lon", geo.get("longitude"))
        else:
            record["geo_latitude"] = geo.get("latitude")
            record["geo_longitude"] = geo.get("longitude")
        record["geo_city"] = geo.get("city_name", "")
        record["geo_region"] = geo.get("region_name", "")
    if "src_ip_identifier" in record and "src_ip" not in record:
        record["src_ip"] = record["src_ip_identifier"]
    return record

def _process_cowrie_chunk(records):
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()
    for c in ["eventid","protocol","day_of_week","geo_country_code","geo_continent"]:
        if c in df.columns: df[c] = df[c].astype("category")
    return df


# =====================================================================
# PyArrow Birleştirme (Şema Uyumlu)
# =====================================================================

def _build_unified_schema(chunk_files):
    column_types = {}
    column_order = []

    for fp in chunk_files:
        schema = pq.read_schema(fp)
        for i in range(len(schema)):
            name = schema.field(i).name
            typ = schema.field(i).type
            if name not in column_types:
                column_types[name] = []
                column_order.append(name)
            column_types[name].append(typ)

    unified_fields = []
    for name in column_order:
        types = column_types[name]
        non_null = [t for t in types if t != pa.null()]
        if non_null:
            chosen = non_null[0]
        else:
            chosen = pa.large_string()
        unified_fields.append(pa.field(name, chosen))

    return pa.schema(unified_fields)


def _cast_table_to_schema(table, target_schema):
    arrays = []
    for field in target_schema:
        if field.name in table.schema.names:
            col = table.column(field.name)
            if col.type == pa.null():
                arrays.append(pa.nulls(len(table), type=field.type))
            elif col.type != field.type:
                try:
                    arrays.append(col.cast(field.type))
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    arrays.append(pa.nulls(len(table), type=field.type))
            else:
                arrays.append(col)
        else:
            arrays.append(pa.nulls(len(table), type=field.type))

    return pa.table(arrays, schema=target_schema)


def _merge_parquet_chunks(tmp_dir, output_path):
    chunk_files = sorted([
        os.path.join(tmp_dir, f)
        for f in os.listdir(tmp_dir) if f.endswith(".parquet")
    ])

    if not chunk_files:
        raise ValueError(f"Chunk bulunamadı: {tmp_dir}")

    print(f"  {len(chunk_files)} chunk → {os.path.basename(output_path)}")

    print(f"  Şema analizi yapılıyor...")
    unified_schema = _build_unified_schema(chunk_files)
    print(f"  Birleşik şema: {len(unified_schema)} sütun")

    print(f"  PyArrow ParquetWriter ile birleştiriliyor...")
    writer = pq.ParquetWriter(output_path, unified_schema,
                               compression=PARQUET_COMPRESSION)
    total_rows = 0

    for i, fp in enumerate(chunk_files):
        table = pq.read_table(fp)
        table = _cast_table_to_schema(table, unified_schema)
        writer.write_table(table)
        total_rows += len(table)
        del table
        gc.collect()

        if (i + 1) % 20 == 0 or (i + 1) == len(chunk_files):
            print(f"    {i+1}/{len(chunk_files)} chunk ({total_rows:,} kayıt)")

    writer.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Birleştirme tamamlandı: {total_rows:,} kayıt")
    return total_rows


# =====================================================================
# ÖZET İSTATİSTİK — Bellek-dostu (sütun-seçimli okuma)
# =====================================================================

def _print_parquet_summary(parquet_path, name):
    """Parquet dosyasından sadece gerekli sütunları okuyarak özet yazdırır."""
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    num_cols = len(pf.schema_arrow)
    schema = pf.schema_arrow
    col_names = [schema.field(i).name for i in range(len(schema))]
    print(f"  {total_rows:,} satır, {num_cols} sütun")

    # --- Tarih aralığı ---
    if "date" in col_names:
        dates = pq.read_table(parquet_path, columns=["date"]).to_pandas()["date"]
        print(f"  Tarih: {dates.min()} → {dates.max()}")
        del dates; gc.collect()

    # --- IP ---
    if "src_ip" in col_names:
        ips = pq.read_table(parquet_path, columns=["src_ip"]).to_pandas()["src_ip"]
        print(f"  IP: {ips.nunique():,}")
        del ips; gc.collect()

    # --- CTU: Protokol ---
    if "proto" in col_names:
        proto = pq.read_table(parquet_path, columns=["proto"]).to_pandas()["proto"]
        print(f"  Protokol:\n{proto.value_counts().to_string()}")
        del proto; gc.collect()

    # --- Cowrie: Event dağılımı ---
    if "eventid" in col_names:
        evts = pq.read_table(parquet_path, columns=["eventid"]).to_pandas()["eventid"]
        print(f"  Olaylar:")
        for e, c in evts.value_counts().items():
            print(f"    {e}: {c:,}")
        del evts; gc.collect()


# =====================================================================
# 1. CTU-Hornet
# =====================================================================

def load_ctu_hornet(data_dir, chunksize=CHUNK_SIZE):
    print("\n" + "=" * 60)
    print(f"CTU-Hornet-65-Niner | Chunk: {chunksize:,}")
    print(f"  Dizin: {data_dir}")
    print("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, "ctu_hornet_processed.parquet")

    # Eğer daha önce başarıyla oluşturulduysa tekrar işleme
    if os.path.exists(output_path):
        try:
            pf = pq.ParquetFile(output_path)
            rows = pf.metadata.num_rows
            if rows > 0:
                print(f"  Mevcut dosya bulundu: {rows:,} kayıt — atlanıyor")
                _print_parquet_summary(output_path, "CTU-Hornet")
                return output_path
        except Exception:
            print(f"  Mevcut dosya bozuk, yeniden oluşturulacak")

    data_files = _find_data_files(data_dir)
    if not data_files: raise FileNotFoundError(f"Dosya bulunamadı: {data_dir}")

    tmp = os.path.join(OUTPUT_DIR, "_tmp_ctu")
    if os.path.exists(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)

    buf, cn, total, skp = [], 0, 0, 0
    t0 = time.time()

    for filepath in data_files:
        for _, line in _open_smart(filepath):
            s = line.strip()
            if not s or not s.startswith("{"): skp += 1; continue
            try: buf.append(json.loads(s)); total += 1
            except json.JSONDecodeError: skp += 1; continue
            if total % 100_000 == 0:
                print(f"  ... {total:,} satır ({total/max(time.time()-t0,1):,.0f}/s)")
            if len(buf) >= chunksize:
                cn += 1; df = _process_ctu_chunk(buf)
                df.to_parquet(os.path.join(tmp, f"c{cn:04d}.parquet"),
                              compression=PARQUET_COMPRESSION, index=False)
                print(f"  Chunk {cn}: {len(df):,} → disk ({total:,} toplam)")
                del df; buf = []; gc.collect()

    if buf:
        cn += 1; df = _process_ctu_chunk(buf)
        df.to_parquet(os.path.join(tmp, f"c{cn:04d}.parquet"),
                      compression=PARQUET_COMPRESSION, index=False)
        print(f"  Chunk {cn}: {len(df):,} → disk (son)")
        del df; buf = []; gc.collect()

    if cn == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"Kayıt bulunamadı: {data_dir}")

    _merge_parquet_chunks(tmp, output_path)
    el = time.time() - t0
    print(f"  Okuma + birleştirme: {el:.0f}s")

    # Özet — tüm DataFrame'i yüklemeden
    _print_parquet_summary(output_path, "CTU-Hornet")

    return output_path  # ← DataFrame değil, dosya yolu


# =====================================================================
# 2. Cowrie
# =====================================================================

def load_cowrie(data_dir, chunksize=CHUNK_SIZE):
    print("\n" + "=" * 60)
    print(f"Cowrie CyberLab | Chunk: {chunksize:,}")
    print(f"  Dizin: {data_dir}")
    print("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, "cowrie_processed.parquet")

    # Eğer daha önce başarıyla oluşturulduysa tekrar işleme
    if os.path.exists(output_path):
        try:
            pf = pq.ParquetFile(output_path)
            rows = pf.metadata.num_rows
            if rows > 0:
                print(f"  Mevcut dosya bulundu: {rows:,} kayıt — atlanıyor")
                _print_parquet_summary(output_path, "Cowrie")
                return output_path
        except Exception:
            print(f"  Mevcut dosya bozuk, yeniden oluşturulacak")

    data_files = _find_data_files(data_dir)
    if not data_files: raise FileNotFoundError(f"Dosya bulunamadı: {data_dir}")

    tmp = os.path.join(OUTPUT_DIR, "_tmp_cowrie")
    if os.path.exists(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)

    buf, cn, total, skp = [], 0, 0, 0
    t0 = time.time()

    for fi, filepath in enumerate(data_files):
        for event in _iter_cowrie_events(filepath):
            eid = event.get("eventid", "")
            if eid not in COWRIE_TARGET_EVENTS: skp += 1; continue
            buf.append(_process_cowrie_record(event))
            total += 1
            if total % 100_000 == 0:
                print(f"  ... {total:,} kayıt ({total/max(time.time()-t0,1):,.0f}/s)")
            if len(buf) >= chunksize:
                cn += 1; df = _process_cowrie_chunk(buf)
                df.to_parquet(os.path.join(tmp, f"c{cn:04d}.parquet"),
                              compression=PARQUET_COMPRESSION, index=False)
                print(f"  Chunk {cn}: {len(df):,} → disk ({total:,} toplam)")
                del df; buf = []; gc.collect()
        if (fi+1) % 20 == 0:
            print(f"  Dosya {fi+1}/{len(data_files)} | {total:,} kayıt | {time.time()-t0:.0f}s")

    if buf:
        cn += 1; df = _process_cowrie_chunk(buf)
        df.to_parquet(os.path.join(tmp, f"c{cn:04d}.parquet"),
                      compression=PARQUET_COMPRESSION, index=False)
        print(f"  Chunk {cn}: {len(df):,} → disk (son)")
        del df; buf = []; gc.collect()

    if cn == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"Kayıt bulunamadı: {data_dir}")

    _merge_parquet_chunks(tmp, output_path)
    el = time.time() - t0
    print(f"  Okuma + birleştirme: {el:.0f}s")

    # Özet — tüm DataFrame'i yüklemeden
    _print_parquet_summary(output_path, "Cowrie")

    return output_path  # ← DataFrame değil, dosya yolu


# =====================================================================
# Yardımcılar — Bellek-dostu versiyonlar
# =====================================================================

def data_quality_report(parquet_path, name):
    """Parquet'ten chunk-chunk okuyarak kalite raporu üretir."""
    print(f"\n{'='*60}\nKALİTE: {name}\n{'='*60}")

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    num_cols = len(pf.schema_arrow)
    schema = pf.schema_arrow
    col_names = [schema.field(i).name for i in range(len(schema))]
    print(f"  {total_rows:,} satır, {num_cols} sütun")

    # Eksik değer sayımı — chunk-chunk oku
    null_counts = {}
    for batch in pf.iter_batches(batch_size=500_000, columns=col_names):
        df_chunk = batch.to_pandas()
        for col in df_chunk.columns:
            null_counts[col] = null_counts.get(col, 0) + int(df_chunk[col].isnull().sum())
        del df_chunk
        gc.collect()

    for col, cnt in null_counts.items():
        if cnt > 0:
            print(f"  Eksik → {col}: {cnt:,}")


def create_unified_format(ctu_path, cowrie_path):
    """
    İki Parquet dosyasından birleşik format oluşturur.
    Sadece gerekli sütunları okur, Cowrie'den sadece connect eventlerini çeker.
    """
    print(f"\n{'='*60}\nBİRLEŞTİRME\n{'='*60}")

    # --- CTU tarafı: sadece gerekli sütunlar ---
    ctu_schema = pq.read_schema(ctu_path)
    ctu_col_names = [ctu_schema.field(i).name for i in range(len(ctu_schema))]

    ctu_cols = ["timestamp", "src_ip", "dst_port", "proto"]
    if "src_country_code" in ctu_col_names:
        ctu_cols.append("src_country_code")

    ctu_sub = pq.read_table(ctu_path, columns=ctu_cols).to_pandas()
    ctu_sub["dataset"] = "CTU-Hornet"
    ctu_sub = ctu_sub.rename(columns={"proto": "protocol"})
    if "src_country_code" in ctu_sub.columns:
        ctu_sub = ctu_sub.rename(columns={"src_country_code": "src_country"})
    ctu_count = len(ctu_sub)

    # --- Cowrie tarafı: sadece connect eventleri, sadece gerekli sütunlar ---
    cowrie_schema = pq.read_schema(cowrie_path)
    cowrie_col_names = [cowrie_schema.field(i).name for i in range(len(cowrie_schema))]

    cowrie_cols = ["timestamp", "src_ip", "eventid"]
    if "geo_country_code" in cowrie_col_names:
        cowrie_cols.append("geo_country_code")

    # Predicate pushdown — sadece connect eventlerini oku
    cowrie_filters = [("eventid", "=", "cowrie.session.connect")]
    try:
        cowrie_sub = pq.read_table(
            cowrie_path, columns=cowrie_cols, filters=cowrie_filters
        ).to_pandas()
    except Exception:
        # Filtre çalışmazsa (dictionary encoding sorunu vb.) elle filtrele
        print("  Uyarı: Predicate pushdown başarısız, eventid sütunu chunk-chunk filtreleniyor...")
        cowrie_parts = []
        pf = pq.ParquetFile(cowrie_path)
        for batch in pf.iter_batches(batch_size=500_000, columns=cowrie_cols):
            df_tmp = batch.to_pandas()
            df_tmp = df_tmp[df_tmp["eventid"] == "cowrie.session.connect"]
            if len(df_tmp) > 0:
                cowrie_parts.append(df_tmp)
            del df_tmp
            gc.collect()
        cowrie_sub = pd.concat(cowrie_parts, ignore_index=True) if cowrie_parts else pd.DataFrame()
        del cowrie_parts; gc.collect()

    cowrie_sub["dataset"] = "Cowrie"
    cowrie_sub = cowrie_sub.drop(columns=["eventid"], errors="ignore")
    if "geo_country_code" in cowrie_sub.columns:
        cowrie_sub = cowrie_sub.rename(columns={"geo_country_code": "src_country"})
    cowrie_count = len(cowrie_sub)

    # --- Birleştir ---
    unified = pd.concat([ctu_sub, cowrie_sub], ignore_index=True)
    del ctu_sub, cowrie_sub; gc.collect()

    # --- Ortak IP hesapla (chunk-chunk) ---
    print(f"  Ortak IP hesaplanıyor...")
    ctu_ips = set(
        pq.read_table(ctu_path, columns=["src_ip"]).to_pandas()["src_ip"].unique()
    )
    # Cowrie IP'lerini chunk-chunk oku (83M satır, tek sütun ~1-2 GB)
    cowrie_ips = set()
    pf = pq.ParquetFile(cowrie_path)
    for batch in pf.iter_batches(batch_size=1_000_000, columns=["src_ip"]):
        cowrie_ips.update(batch.to_pandas()["src_ip"].unique())
        gc.collect()

    common_ips = ctu_ips & cowrie_ips
    del ctu_ips, cowrie_ips; gc.collect()

    print(f"  CTU: {ctu_count:,} + Cowrie: {cowrie_count:,} = {len(unified):,} | Ortak IP: {len(common_ips):,}")
    return unified, common_ips


def generate_summary_statistics(ctu_path, cowrie_path):
    """Parquet dosyalarından özet istatistik üretir — bellek dostu."""
    print(f"\n{'='*60}\nİSTATİSTİK\n{'='*60}")

    # --- CTU: Satır sayısı, IP sayısı, top portlar ---
    ctu_pf = pq.ParquetFile(ctu_path)
    ctu_rows = ctu_pf.metadata.num_rows

    ctu_ips = pq.read_table(ctu_path, columns=["src_ip"]).to_pandas()["src_ip"]
    print(f"  CTU: {ctu_rows:,} flow, {ctu_ips.nunique():,} IP")
    del ctu_ips; gc.collect()

    # Top portlar — chunk-chunk say
    port_counts = {}
    for batch in ctu_pf.iter_batches(batch_size=500_000, columns=["dst_port"]):
        for port, cnt in batch.to_pandas()["dst_port"].value_counts().items():
            port_counts[port] = port_counts.get(port, 0) + cnt
        gc.collect()
    top_ports = sorted(port_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for p, c in top_ports:
        print(f"    Port {p}: {c:,}")
    del port_counts; gc.collect()

    # --- Cowrie: Oturum sayısı, IP sayısı, top kullanıcılar ---
    cowrie_schema = pq.read_schema(cowrie_path)
    cowrie_col_names = [cowrie_schema.field(i).name for i in range(len(cowrie_schema))]

    # Oturum sayısı (connect eventleri)
    session_count = 0
    cowrie_pf = pq.ParquetFile(cowrie_path)
    for batch in cowrie_pf.iter_batches(batch_size=500_000, columns=["eventid"]):
        session_count += int((batch.to_pandas()["eventid"] == "cowrie.session.connect").sum())
        gc.collect()

    # IP sayısı
    cowrie_ips = set()
    for batch in cowrie_pf.iter_batches(batch_size=1_000_000, columns=["src_ip"]):
        cowrie_ips.update(batch.to_pandas()["src_ip"].unique())
        gc.collect()
    print(f"\n  Cowrie: {session_count:,} oturum, {len(cowrie_ips):,} IP")
    del cowrie_ips; gc.collect()

    # Top kullanıcı adları
    if "username" in cowrie_col_names:
        user_counts = {}
        login_cols = ["eventid", "username"]
        for batch in cowrie_pf.iter_batches(batch_size=500_000, columns=login_cols):
            df_tmp = batch.to_pandas()
            logins = df_tmp[df_tmp["eventid"].str.contains("login", na=False)]
            logins = logins.dropna(subset=["username"])
            for u, c in logins["username"].value_counts().items():
                user_counts[u] = user_counts.get(u, 0) + c
            del df_tmp, logins; gc.collect()
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top kullanıcı: {dict(top_users)}")
        del user_counts; gc.collect()


# =====================================================================
# GeoLite2 Zenginleştirme — chunk-chunk
# =====================================================================

def enrich_ctu_geolite2(ctu_path, geolite2_path):
    """CTU Parquet'i GeoLite2 ile zenginleştirir — chunk-chunk yazar."""
    schema = pq.read_schema(ctu_path)
    col_names = [schema.field(i).name for i in range(len(schema))]
    if "src_country_code" in col_names:
        print(f"  GeoLite2 zaten uygulanmış — atlanıyor")
        return ctu_path

    try:
        from geolocation import GeoLocator
    except ImportError:
        print(f"  UYARI: geolocation modülü bulunamadı — GeoLite2 atlanıyor")
        return ctu_path

    geo = GeoLocator(geolite2_path)
    if not geo.primary:
        print(f"  UYARI: GeoLite2 veritabanı yüklenemedi — atlanıyor")
        geo.close()
        return ctu_path

    print(f"\n  GeoLite2 zenginleştirme (chunk-chunk)...")

    enriched_path = ctu_path + ".enriched"
    pf = pq.ParquetFile(ctu_path)
    writer = None
    total = 0

    for batch in pf.iter_batches(batch_size=500_000):
        df_chunk = batch.to_pandas()
        df_chunk = geo.enrich_dataframe(df_chunk, ip_column="src_ip", prefix="geo_")
        df_chunk["src_country_code"] = df_chunk.get("geo_country_code", "")
        df_chunk["src_country_name"] = df_chunk.get("geo_country_name", "")

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(enriched_path, table.schema,
                                       compression=PARQUET_COMPRESSION)
        else:
            # Şema uyumu
            table = _cast_table_to_schema(table, writer.schema)
        writer.write_table(table)
        total += len(df_chunk)
        del df_chunk, table; gc.collect()
        if total % 2_000_000 == 0:
            print(f"    {total:,} kayıt zenginleştirildi")

    if writer:
        writer.close()

    geo.save_cache(os.path.join(OUTPUT_DIR, "geo_cache.json"))
    geo.close()

    # Eski dosyayı değiştir
    os.replace(enriched_path, ctu_path)
    print(f"  GeoLite2 tamamlandı: {total:,} kayıt")
    return ctu_path


# =====================================================================
# ANA
# =====================================================================

if __name__ == "__main__":
    print("╔" + "═"*58 + "╗")
    print("║  HAFTA 1: VERİ ENTEGRASYonu (v7 — Bellek-Dostu)          ║")
    print("╚" + "═"*58 + "╝")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Cowrie → dosya yolu döner
    cowrie_path = load_cowrie(COWRIE_DIR)

    # 2. CTU-Hornet → dosya yolu döner
    ctu_path = load_ctu_hornet(CTU_HORNET_DIR)

    # 3. Kalite raporu — chunk-chunk
    data_quality_report(ctu_path, "CTU-Hornet")
    data_quality_report(cowrie_path, "Cowrie")

    # 4. GeoLite2 zenginleştirme — chunk-chunk
    ctu_path = enrich_ctu_geolite2(ctu_path, GEOLITE2_PATH)

    # 5. Birleştir & İstatistik
    unified_df, common_ips = create_unified_format(ctu_path, cowrie_path)
    generate_summary_statistics(ctu_path, cowrie_path)

    # 6. Kaydet
    print(f"\n  Kaydediliyor...")
    unified_df.to_parquet(f"{OUTPUT_DIR}/unified_dataset.parquet",
                          compression=PARQUET_COMPRESSION, index=False)
    print(f"    unified_dataset.parquet ✓")
    del unified_df; gc.collect()

    if common_ips:
        pd.DataFrame({"ip": list(common_ips)}).to_csv(
            f"{OUTPUT_DIR}/common_ips.csv", index=False
        )
        print(f"    common_ips.csv ✓ ({len(common_ips):,} IP)")
    del common_ips; gc.collect()

    print(f"\n{'='*60}")
    print(f"Tamamlandı! → {OUTPUT_DIR}/")
    print(f"  ctu_hornet_processed.parquet")
    print(f"  cowrie_processed.parquet")
    print(f"  unified_dataset.parquet")
    print(f"{'='*60}")