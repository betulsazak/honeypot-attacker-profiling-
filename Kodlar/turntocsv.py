"""
Parquet → CSV Dönüştürücü
==========================
output/ altındaki parquet dosyalarını CSV'ye çevirir.
Büyük dosyaları chunk-chunk okuyarak bellek dostu çalışır.
"""

import os
import gc
import time
import pyarrow.parquet as pq

OUTPUT_DIR = "/mnt/c/TEZ_PROJEEE/output"
BATCH_SIZE = 500_000


def parquet_to_csv(parquet_path, csv_path):
    """Parquet dosyasını chunk-chunk okuyarak CSV'ye yazar."""
    pf = pq.ParquetFile(parquet_path)
    rows = pf.metadata.num_rows
    print(f"  {os.path.basename(parquet_path)} ({rows:,} satır) → {os.path.basename(csv_path)}")

    first = True
    written = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        df.to_csv(csv_path, mode="w" if first else "a", index=False, header=first)
        written += len(df)
        first = False
        del df; gc.collect()
        if written % 2_000_000 < BATCH_SIZE:
            print(f"    ... {written:,} / {rows:,}")

    size_mb = os.path.getsize(csv_path) / 1e6
    print(f"    Tamamlandı: {size_mb:.1f} MB")


if __name__ == "__main__":
    print("╔" + "═" * 50 + "╗")
    print("║  PARQUET → CSV DÖNÜŞTÜRÜCÜ                    ║")
    print("╚" + "═" * 50 + "╝")
    t0 = time.time()

    targets = [
        "ctu_hornet_processed.parquet",
        "cowrie_processed.parquet",
        "unified_dataset.parquet",
    ]

    for fname in targets:
        pq_path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(pq_path):
            print(f"  {fname} bulunamadı — atlanıyor")
            continue
        csv_path = pq_path.replace(".parquet", ".csv")
        parquet_to_csv(pq_path, csv_path)

    elapsed = time.time() - t0
    print(f"\nTamamlandı — {elapsed:.0f}s")
    print(f"CSV dosyaları: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".csv"):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
            print(f"  {f}  ({size:.1f} MB)")