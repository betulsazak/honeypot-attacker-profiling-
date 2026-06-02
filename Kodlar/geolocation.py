"""
IP Geolokasyon Zenginleştirme Modülü
=====================================
MaxMind GeoLite2 (offline, hızlı) + ip-api.com (online, fallback)

Kullanım:
    1. https://www.maxmind.com/en/geolite2/signup adresinden ücretsiz hesap aç
    2. License key oluştur
    3. GeoLite2-City.mmdb dosyasını indir:
       curl -L -u ACCOUNT_ID:LICENSE_KEY \
         "https://download.maxmind.com/geoip/databases/GeoLite2-City/download?suffix=tar.gz" \
         -o GeoLite2-City.tar.gz
       tar -xzf GeoLite2-City.tar.gz
       cp GeoLite2-City_*/GeoLite2-City.mmdb data/geolite2/
    4. Bu modülü import et ve kullan:
       from geolocation import GeoLocator
       geo = GeoLocator("data/geolite2/GeoLite2-City.mmdb")
       info = geo.lookup("8.8.8.8")

Yazar: [Tez Sahibi]
Tarih: 2025
"""

import os
import json
import time
import logging
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# =====================================================================
# Veri Yapısı
# =====================================================================

@dataclass
class GeoInfo:
    """Bir IP adresi için coğrafi konum bilgisi."""
    ip: str
    country_code: str = "XX"
    country_name: str = "Unknown"
    city: str = "Unknown"
    latitude: float = 0.0
    longitude: float = 0.0
    continent_code: str = "XX"
    continent_name: str = "Unknown"
    region: str = "Unknown"
    asn: int = 0
    as_org: str = "Unknown"
    accuracy_radius: int = 0  # km cinsinden

    def to_dict(self):
        return asdict(self)


# =====================================================================
# MaxMind GeoLite2 Backend
# =====================================================================

class MaxMindBackend:
    """
    MaxMind GeoLite2 City veritabanı ile offline IP geolokasyon.
    
    Avantajları:
    - Çok hızlı: ~50,000 sorgu/saniye
    - Rate limit yok
    - İnternet bağlantısı gerektirmez
    - City düzeyinde çözünürlük
    
    Gereksinim:
    - pip install geoip2
    - GeoLite2-City.mmdb dosyası
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        try:
            import geoip2.database
            if os.path.exists(self.db_path):
                self.reader = geoip2.database.Reader(self.db_path)
                # Veritabanı bilgilerini yazdır
                meta = self.reader.metadata()
                logger.info(
                    f"GeoLite2 yuklenindi: {meta.database_type}, "
                    f"build={meta.build_epoch}, "
                    f"node_count={meta.node_count}"
                )
                print(f"  [GeoLite2] Veritabani yuklenindi: {self.db_path}")
                print(f"  [GeoLite2] Tip: {meta.database_type}")
            else:
                logger.warning(f"GeoLite2 veritabani bulunamadi: {self.db_path}")
                print(f"  [GeoLite2] UYARI: Veritabani bulunamadi: {self.db_path}")
                print(f"  [GeoLite2] Indirme talimatlari icin README.md dosyasina bakin.")
        except ImportError:
            logger.error("geoip2 kutuphanesi yuklu degil: pip install geoip2")
            print("  [GeoLite2] HATA: geoip2 yuklu degil. Calistirin: pip install geoip2")

    def is_available(self) -> bool:
        return self.reader is not None

    def lookup(self, ip: str) -> Optional[GeoInfo]:
        if not self.reader:
            return None
        try:
            response = self.reader.city(ip)
            return GeoInfo(
                ip=ip,
                country_code=response.country.iso_code or "XX",
                country_name=response.country.name or "Unknown",
                city=response.city.name or "Unknown",
                latitude=response.location.latitude or 0.0,
                longitude=response.location.longitude or 0.0,
                continent_code=response.continent.code or "XX",
                continent_name=response.continent.name or "Unknown",
                region=(response.subdivisions.most_specific.name
                        if response.subdivisions else "Unknown"),
                asn=0,  # ASN icin ayri veritabani gerekli
                as_org="Unknown",
                accuracy_radius=response.location.accuracy_radius or 0,
            )
        except Exception as e:
            # Ozel IP araliklari, bilinmeyen IP'ler vb.
            logger.debug(f"GeoLite2 lookup basarisiz {ip}: {e}")
            return GeoInfo(ip=ip)

    def lookup_batch(self, ip_list: list) -> dict:
        """Toplu IP sorgusu (MaxMind zaten cok hizli, loop yeterli)."""
        results = {}
        for ip in ip_list:
            result = self.lookup(ip)
            if result:
                results[ip] = result
        return results

    def close(self):
        if self.reader:
            self.reader.close()


# =====================================================================
# ip-api.com Fallback Backend
# =====================================================================

class IpApiBackend:
    """
    ip-api.com batch API ile online IP geolokasyon.
    
    Sinirliliklari:
    - Ucretsiz katman: 45 istek/dakika
    - Batch: 100 IP/istek
    - 294,000 IP icin ~49 dakika surer
    - Internet baglantisi gerektirir
    
    Yalnizca MaxMind GeoLite2 mevcut degilse fallback olarak kullanilir.
    """

    BASE_URL = "http://ip-api.com/batch"
    RATE_LIMIT_DELAY = 1.5  # saniye (45 istek/dakika = 1.33s, guvenlik payiyla 1.5s)
    BATCH_SIZE = 100

    def __init__(self):
        self._check_requests()

    def _check_requests(self):
        try:
            import requests
            self.requests = requests
            print("  [ip-api.com] Fallback backend hazir")
        except ImportError:
            self.requests = None
            print("  [ip-api.com] UYARI: requests yuklu degil: pip install requests")

    def is_available(self) -> bool:
        return self.requests is not None

    def lookup(self, ip: str) -> Optional[GeoInfo]:
        if not self.requests:
            return None
        try:
            resp = self.requests.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "country,countryCode,city,lat,lon,regionName,continent,continentCode,as,org,isp"},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "fail":
                    return GeoInfo(ip=ip)
                return GeoInfo(
                    ip=ip,
                    country_code=data.get("countryCode", "XX"),
                    country_name=data.get("country", "Unknown"),
                    city=data.get("city", "Unknown"),
                    latitude=data.get("lat", 0.0),
                    longitude=data.get("lon", 0.0),
                    continent_code=data.get("continentCode", "XX"),
                    continent_name=data.get("continent", "Unknown"),
                    region=data.get("regionName", "Unknown"),
                    as_org=data.get("org", "Unknown"),
                )
        except Exception as e:
            logger.debug(f"ip-api.com lookup basarisiz {ip}: {e}")
        return GeoInfo(ip=ip)

    def lookup_batch(self, ip_list: list) -> dict:
        """Toplu IP sorgusu (batch API ile)."""
        if not self.requests:
            return {}

        results = {}
        total = len(ip_list)
        processed = 0

        for i in range(0, total, self.BATCH_SIZE):
            batch = ip_list[i:i + self.BATCH_SIZE]
            payload = [
                {"query": ip, "fields": "country,countryCode,city,lat,lon,regionName,continent,continentCode,as,org"}
                for ip in batch
            ]

            try:
                resp = self.requests.post(self.BASE_URL, json=payload, timeout=15)
                if resp.status_code == 200:
                    for item in resp.json():
                        ip = item.get("query", "")
                        if item.get("status") == "success":
                            results[ip] = GeoInfo(
                                ip=ip,
                                country_code=item.get("countryCode", "XX"),
                                country_name=item.get("country", "Unknown"),
                                city=item.get("city", "Unknown"),
                                latitude=item.get("lat", 0.0),
                                longitude=item.get("lon", 0.0),
                                continent_code=item.get("continentCode", "XX"),
                                continent_name=item.get("continent", "Unknown"),
                                region=item.get("regionName", "Unknown"),
                                as_org=item.get("org", "Unknown"),
                            )
                        else:
                            results[ip] = GeoInfo(ip=ip)
                elif resp.status_code == 429:
                    logger.warning("ip-api.com rate limit asildi, bekleniyor...")
                    print(f"  [ip-api.com] Rate limit! 60 saniye bekleniyor...")
                    time.sleep(60)
                    continue  # Ayni batch'i tekrar dene
            except Exception as e:
                logger.error(f"ip-api.com batch hatasi: {e}")

            processed += len(batch)
            if processed % 1000 == 0 or processed == total:
                print(f"  [ip-api.com] Ilerleme: {processed}/{total} IP ({processed/total*100:.1f}%)")

            # Rate limiting
            time.sleep(self.RATE_LIMIT_DELAY)

        return results


# =====================================================================
# Birlesik GeoLocator (Ana Arayuz)
# =====================================================================

class GeoLocator:
    """
    IP geolokasyon ana sinifi.
    
    Oncelik sirasi:
    1. MaxMind GeoLite2 (offline, hizli) - varsa
    2. ip-api.com (online, yavas) - fallback
    3. Cache (tekrar sorgulamayi onle)
    
    Kullanim:
        geo = GeoLocator("data/geolite2/GeoLite2-City.mmdb")
        
        # Tekli sorgu
        info = geo.lookup("8.8.8.8")
        print(info.country_name)  # "United States"
        
        # Toplu sorgu (DataFrame ile)
        df = geo.enrich_dataframe(df, ip_column="src_ip")
    """

    def __init__(self, geolite2_path: str = "data/geolite2/GeoLite2-City.mmdb"):
        print("\n" + "=" * 60)
        print("IP Geolokasyon Modulu Baslatiliyor")
        print("=" * 60)

        self.cache = {}  # IP -> GeoInfo cache
        self.maxmind = MaxMindBackend(geolite2_path)
        self.ipapi = IpApiBackend()

        if self.maxmind.is_available():
            self.primary = "maxmind"
            print(f"  Birincil backend: MaxMind GeoLite2")
        elif self.ipapi.is_available():
            self.primary = "ipapi"
            print(f"  Birincil backend: ip-api.com (MaxMind bulunamadi)")
            print(f"  UYARI: Buyuk veri setleri icin MaxMind oneriliyor!")
        else:
            self.primary = None
            print(f"  HATA: Hicbir geolokasyon backend'i kullanilabilir degil!")

    def lookup(self, ip: str) -> GeoInfo:
        """Tekli IP sorgusu (cache'li)."""
        # Ozel IP araliklarini atla
        if ip.startswith(("10.", "172.16.", "192.168.", "127.", "0.")):
            return GeoInfo(ip=ip, country_name="Private", country_code="--")

        # Cache kontrol
        if ip in self.cache:
            return self.cache[ip]

        # MaxMind (birincil)
        result = None
        if self.maxmind.is_available():
            result = self.maxmind.lookup(ip)

        # ip-api.com (fallback)
        if (result is None or result.country_code == "XX") and self.ipapi.is_available():
            result = self.ipapi.lookup(ip)

        if result is None:
            result = GeoInfo(ip=ip)

        self.cache[ip] = result
        return result

    def lookup_batch(self, ip_list: list) -> dict:
        """
        Toplu IP sorgusu.
        Oncesinde cache'ten bakar, sonra toplu sorgu yapar.
        """
        # Benzersiz IP'leri al
        unique_ips = list(set(ip_list))

        # Ozel IP'leri ayir
        public_ips = [ip for ip in unique_ips
                      if not ip.startswith(("10.", "172.16.", "192.168.", "127.", "0."))]
        private_ips = [ip for ip in unique_ips
                       if ip.startswith(("10.", "172.16.", "192.168.", "127.", "0."))]

        # Ozel IP'leri isaretle
        for ip in private_ips:
            self.cache[ip] = GeoInfo(ip=ip, country_name="Private", country_code="--")

        # Cache'te olmayanlar
        uncached = [ip for ip in public_ips if ip not in self.cache]

        if len(uncached) == 0:
            print(f"  Tum IP'ler cache'te ({len(unique_ips)} IP)")
            return self.cache

        print(f"  Sorgulanacak IP sayisi: {len(uncached)} / {len(unique_ips)} benzersiz")

        # MaxMind (hizli, toplu)
        if self.maxmind.is_available():
            print(f"  [MaxMind] Toplu sorgu baslatiliyor...")
            t0 = time.time()
            results = self.maxmind.lookup_batch(uncached)
            elapsed = time.time() - t0
            self.cache.update(results)
            print(f"  [MaxMind] {len(results)} IP sorguland, sure: {elapsed:.2f}s "
                  f"({len(results)/max(elapsed, 0.001):.0f} IP/s)")

            # MaxMind'da bulunamayanlar
            still_missing = [ip for ip in uncached if ip not in self.cache or self.cache[ip].country_code == "XX"]
            if still_missing and self.ipapi.is_available():
                print(f"  [ip-api.com] {len(still_missing)} IP icin fallback sorgusu...")
                fallback_results = self.ipapi.lookup_batch(still_missing)
                self.cache.update(fallback_results)

        # Sadece ip-api.com
        elif self.ipapi.is_available():
            print(f"  [ip-api.com] Toplu sorgu baslatiliyor ({len(uncached)} IP)...")
            print(f"  UYARI: Tahmini sure: ~{len(uncached) / self.ipapi.BATCH_SIZE * self.ipapi.RATE_LIMIT_DELAY / 60:.1f} dakika")
            results = self.ipapi.lookup_batch(uncached)
            self.cache.update(results)

        return self.cache

    def enrich_dataframe(self, df, ip_column="src_ip", prefix="geo_"):
        """
        pandas DataFrame'e geolokasyon sütunları ekler.
        
        Parametre:
            df: pandas DataFrame
            ip_column: IP adresi iceren sutun adi
            prefix: Eklenen sutunlarin on eki
            
        Donus:
            Zenginlestirilmis DataFrame (yeni sutunlar eklenmis)
        """
        import pandas as pd

        print(f"\n  DataFrame zenginlestirme: {len(df)} satir, IP sutunu: '{ip_column}'")

        # Benzersiz IP'leri sorgula
        unique_ips = df[ip_column].dropna().unique().tolist()
        self.lookup_batch(unique_ips)

        # Sonuclari DataFrame'e ekle
        geo_data = []
        for ip in df[ip_column]:
            info = self.cache.get(ip, GeoInfo(ip=str(ip)))
            geo_data.append({
                f"{prefix}country_code": info.country_code,
                f"{prefix}country_name": info.country_name,
                f"{prefix}city": info.city,
                f"{prefix}latitude": info.latitude,
                f"{prefix}longitude": info.longitude,
                f"{prefix}continent_code": info.continent_code,
                f"{prefix}continent_name": info.continent_name,
                f"{prefix}region": info.region,
                f"{prefix}accuracy_radius_km": info.accuracy_radius,
            })

        geo_df = pd.DataFrame(geo_data)
        result = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

        # Ozet istatistikler
        resolved = sum(1 for g in geo_data if g[f"{prefix}country_code"] not in ("XX", "--"))
        total = len(geo_data)
        print(f"  Cozumlenen IP orani: {resolved}/{total} ({resolved/max(total,1)*100:.1f}%)")

        top_countries = result[f"{prefix}country_name"].value_counts().head(5)
        print(f"  En cok saldiri yapan ulkeler (Top 5):")
        for country, count in top_countries.items():
            print(f"    {country}: {count:,}")

        return result

    def get_stats(self) -> dict:
        """Cache istatistiklerini dondurur."""
        total = len(self.cache)
        resolved = sum(1 for g in self.cache.values() if g.country_code not in ("XX", "--"))
        countries = set(g.country_code for g in self.cache.values() if g.country_code not in ("XX", "--"))
        return {
            "total_cached": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "unique_countries": len(countries),
            "primary_backend": self.primary,
        }

    def save_cache(self, filepath: str):
        """Cache'i JSON dosyasina kaydeder (tekrar kullanim icin)."""
        data = {ip: info.to_dict() for ip, info in self.cache.items()}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Cache kaydedildi: {filepath} ({len(data)} IP)")

    def load_cache(self, filepath: str):
        """Onceden kaydedilmis cache'i yukler."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            for ip, info_dict in data.items():
                self.cache[ip] = GeoInfo(**info_dict)
            print(f"  Cache yuklendi: {filepath} ({len(data)} IP)")
        else:
            print(f"  Cache dosyasi bulunamadi: {filepath}")

    def close(self):
        self.maxmind.close()


# =====================================================================
# TEST VE DEMO
# =====================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("IP GEOLOKASYON MODULU - TEST")
    print("=" * 60)

    # GeoLite2 yolunu belirle
    GEOLITE2_PATH = "data/geolite2/GeoLite2-City.mmdb"

    # Komut satiri argumani olarak yol verilebilir
    if len(sys.argv) > 1:
        GEOLITE2_PATH = sys.argv[1]

    geo = GeoLocator(GEOLITE2_PATH)

    # Test IP'leri
    test_ips = [
        "8.8.8.8",        # Google DNS (ABD)
        "1.1.1.1",        # Cloudflare (Avustralya)
        "114.114.114.114", # Cin DNS
        "77.88.8.8",       # Yandex (Rusya)
        "103.224.182.250", # Hindistan
        "192.168.1.1",     # Ozel (Private)
        "185.220.101.1",   # Tor cikis dugumu
    ]

    print(f"\n--- Tekli Sorgu Testi ---")
    for ip in test_ips:
        info = geo.lookup(ip)
        print(f"  {ip:20s} -> {info.country_name} ({info.country_code}), "
              f"{info.city}, [{info.latitude:.2f}, {info.longitude:.2f}]")

    # DataFrame testi
    print(f"\n--- DataFrame Zenginlestirme Testi ---")
    try:
        import pandas as pd

        # Ornek DataFrame
        sample_df = pd.DataFrame({
            "src_ip": test_ips * 3,
            "dst_port": [22, 80, 443, 23, 3389, 445, 8080] * 3,
        })

        enriched = geo.enrich_dataframe(sample_df, ip_column="src_ip")
        print(f"\n  Zenginlestirilmis sutunlar: {[c for c in enriched.columns if c.startswith('geo_')]}")
        print(f"\n  Ornek satirlar:")
        print(enriched[["src_ip", "geo_country_name", "geo_city", "geo_latitude", "geo_longitude"]].head(7).to_string(index=False))
    except ImportError:
        print("  pandas yuklu degil, DataFrame testi atlandi")

    # Cache kaydet
    geo.save_cache("output/geo_cache.json")

    # Istatistikler
    stats = geo.get_stats()
    print(f"\n--- Istatistikler ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    geo.close()
    print(f"\n{'=' * 60}")
    print("Test tamamlandi!")

    if not geo.maxmind.is_available():
        print(f"\n{'!' * 60}")
        print("ONEMLI: MaxMind GeoLite2 veritabani bulunamadi!")
        print("Indirme adimlari:")
        print("  1. https://www.maxmind.com/en/geolite2/signup adresinden ucretsiz hesap ac")
        print("  2. Hesap panelinden License Key olustur")
        print("  3. Asagidaki komutu calistir:")
        print(f'     curl -L -u ACCOUNT_ID:LICENSE_KEY \\')
        print(f'       "https://download.maxmind.com/geoip/databases/GeoLite2-City/download?suffix=tar.gz" \\')
        print(f'       -o GeoLite2-City.tar.gz')
        print(f'     tar -xzf GeoLite2-City.tar.gz')
        print(f'     cp GeoLite2-City_*/GeoLite2-City.mmdb {GEOLITE2_PATH}')
        print(f"{'!' * 60}")