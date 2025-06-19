import ee
import os
import json
import time
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta

class Sentinel2Processor:
    def __init__(self):
        """Sentinel-2 verilerini işlemek için sınıf"""
        # Earth Engine'i başlat
        ee.Initialize(project='agr-46')  # Google Earth Engine projesi
        
        # Sentinel-2 bantları
        self.bands = {
            'B1': 'Coastal aerosol',
            'B2': 'Blue',
            'B3': 'Green',
            'B4': 'Red',
            'B5': 'Red edge 1',
            'B6': 'Red edge 2',
            'B7': 'Red edge 3',
            'B8': 'NIR',
            'B8A': 'Red edge 4',
            'B9': 'Water vapor',
            'B11': 'SWIR 1',
            'B12': 'SWIR 2'
        }
        
        # CDL'de mısır için kod
        self.corn_code = 1  # CDL'de mısır için kod
        
    def get_iowa_counties(self):
        """Iowa ilçelerini alır"""
        # Iowa ilçelerini al
        counties = ee.FeatureCollection('TIGER/2018/Counties') \
            .filter(ee.Filter.eq('STATEFP', '19'))  # Iowa FIPS kodu
        
        # GeoPandas için dönüştür
        counties_list = counties.getInfo()['features']
        counties_df = pd.DataFrame([{
            'NAME': f['properties']['NAME'],
            'COUNTYFP': f['properties']['COUNTYFP'],
            'geometry': f['geometry']
        } for f in counties_list])
        
        return counties_df
    
    def get_cdl_mask(self, year, county_geom):
        """
        Belirli bir yıl için CDL maskesini alır
        
        Args:
            year: Yıl
            county_geom: İlçe geometrisi
        """
        try:
            # CDL koleksiyonunu al
            cdl = ee.Image(f'USDA/NASS/CDL/{year}')
            
            # Mısır maskesini oluştur (1: Corn)
            corn_mask = cdl.select('cropland').eq(1)
            
            # İlçe sınırlarına göre kırp
            corn_mask = corn_mask.clip(county_geom)
            
            return corn_mask
            
        except Exception as e:
            print(f"CDL veri çekme hatası: {str(e)}")
            return None
    
    def get_sentinel_data(self, county_geom, start_date, end_date, year):
        """
        Belirli bir ilçe için Sentinel-2 verilerini alır
        
        Args:
            county_geom: İlçe geometrisi
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            year: Yıl (CDL için)
        """
        try:
            # GeoPandas geometrisini GEE geometry'ye çevir
            gee_geom = ee.Geometry.Polygon(county_geom['coordinates'])
            
            # CDL maskesini al
            corn_mask = self.get_cdl_mask(year, gee_geom)
            if corn_mask is None:
                print(f"    Uyarı: {year} için CDL verisi bulunamadı")
                return None
            
            # Sentinel-2 koleksiyonunu al
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(gee_geom) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # Koleksiyon boyutunu kontrol et
            size = collection.size().getInfo()
            if size == 0:
                print(f"    Uyarı: {start_date} - {end_date} arasında görüntü bulunamadı")
                return None
            
            # Görüntü bilgilerini al
            image_info = collection.aggregate_array('system:index').getInfo()
            print(f"    Bulunan görüntüler: {image_info}")
            
            # Her görüntü için işlem yap
            results = []
            image_list = collection.toList(size)
            
            for i in range(size):
                # Görüntüyü al
                image = ee.Image(image_list.get(i))
                
                # Görüntü bilgilerini al
                image_id = image.get('system:index').getInfo()
                timestamp = image.get('system:time_start').getInfo()
                date = datetime.fromtimestamp(timestamp/1000)
                
                print(f"    İşleniyor: {image_id} - {date}")
                
                # Bantları seç
                bands = list(self.bands.keys())
                image = image.select(bands)
                
                # Görüntüyü mısır maskesine göre maskele
                masked_image = image.updateMask(corn_mask)
                
                # İlçe içindeki ortalama değerleri hesapla
                stats = masked_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=gee_geom,
                    scale=10,  # Sentinel-2 çözünürlüğü
                    maxPixels=1e13
                ).getInfo()
                
                if stats:  # Boş olmayan sonuçları al
                    results.append(stats)
            
            if not results:
                return None
            
            # DataFrame oluştur
            df = pd.DataFrame(results)
            
            # Tarihleri ekle
            dates = collection.aggregate_array('system:time_start').getInfo()
            df.index = pd.to_datetime([datetime.fromtimestamp(d/1000) for d in dates])
            
            return df
            
        except Exception as e:
            print(f"    Hata: {str(e)}")
            return None
    
    def process_all_counties(self, start_year=2018, end_year=2023):
        """
        Tüm Iowa ilçeleri için Sentinel-2 verilerini işler
        
        Args:
            start_year: Başlangıç yılı
            end_year: Bitiş yılı
        """
        # Çıktı dizinini oluştur
        output_dir = 'data/satellite/processed'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Çıktı dizini: {os.path.abspath(output_dir)}")
        
        # CSV dosyasını oluştur ve başlıkları yaz
        output_file = os.path.join(output_dir, f'iowa_corn_sentinel2_{start_year}_{end_year}.csv')
        
        # Sütun sırasını düzenle
        columns = [
            'county_name',    # İlçe adı
            'year',           # Yıl
            'month',          # Ay
            'date',           # Tarih
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6',  # Mavi bantlar
            'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'   # Kızılötesi bantlar
        ]
        
        with open(output_file, 'w') as f:
            f.write(','.join(columns) + '\n')
        
        # İlçeleri al
        counties = self.get_iowa_counties()
        print(f"Toplam {len(counties)} ilçe bulundu")
        
        # Her ilçe için verileri al
        for idx, county in counties.iterrows():
            print(f"\nİşleniyor: {county['NAME']} County ({idx+1}/{len(counties)})")
            
            # Her yıl için verileri al
            for year in range(start_year, end_year + 1):
                # Mısır ekim dönemi için aylık tarih aralıkları
                growing_season = [
                    (f"{year}-04-01", f"{year}-04-30", "Nisan"),    # Ekim öncesi
                    (f"{year}-05-01", f"{year}-05-31", "Mayıs"),    # Ekim
                    (f"{year}-06-01", f"{year}-06-30", "Haziran"),  # Çimlenme
                    (f"{year}-07-01", f"{year}-07-31", "Temmuz"),   # Gelişim
                    (f"{year}-08-01", f"{year}-08-31", "Ağustos"),  # Gelişim
                    (f"{year}-09-01", f"{year}-09-30", "Eylül"),    # Olgunlaşma
                    (f"{year}-10-01", f"{year}-10-31", "Ekim")      # Hasat
                ]
                
                for start_date, end_date, month in growing_season:
                    print(f"  Dönem: {month} {year}")
                    
                    try:
                        # Sentinel-2 verilerini al
                        county_data = self.get_sentinel_data(
                            county['geometry'],
                            start_date,
                            end_date,
                            year
                        )
                        
                        if county_data is not None and not county_data.empty:
                            # İlçe bilgilerini ekle
                            county_data['county_name'] = county['NAME']
                            county_data['year'] = year
                            county_data['month'] = month
                            county_data['date'] = start_date
                            
                            # Sütun sırasını düzenle
                            county_data = county_data[columns]
                            
                            # Verileri CSV'ye yaz
                            county_data.to_csv(output_file, mode='a', header=False, index=False)
                            print(f"    Veri kaydedildi: {len(county_data)} görüntü")
                        else:
                            print(f"    Uyarı: {month} {year} için veri bulunamadı")
                            
                    except Exception as e:
                        print(f"    Hata: {str(e)}")
                        continue
        
        print(f"\nTüm veriler kaydedildi: {output_file}")
        
        # Son durumu kontrol et
        try:
            final_df = pd.read_csv(output_file)
            print(f"Toplam {len(final_df)} satır veri işlendi")
        except Exception as e:
            print(f"Uyarı: CSV dosyası okunamadı: {str(e)}")
            return None

if __name__ == "__main__":
    # İşlemciyi oluştur
    processor = Sentinel2Processor()
    
    # Verileri işle
    processor.process_all_counties() 