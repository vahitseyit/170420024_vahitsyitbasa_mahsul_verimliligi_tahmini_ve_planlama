import ee

# Earth Engine oturumu başlat
ee.Initialize()

# 1. Iowa eyalet sınırlarını al (TIGER dataset)
iowa = ee.FeatureCollection("TIGER/2018/States") \
          .filter(ee.Filter.eq('NAME', 'Iowa'))

# 2. USDA Cropland Data Layer'dan 2021 verisini çek
cdl = ee.ImageCollection("USDA/NASS/CDL") \
          .filterDate("2021-01-01", "2021-12-31") \
          .first() \
          .select("cropland")

# 3. CDL'de mısırın sınıf değeri 1
corn_mask = cdl.eq(1)

# 4. Maskeyi uygulayarak sadece mısır tarlalarını al
corn_only = cdl.updateMask(corn_mask)

# 5. Görüntüyü Iowa ile sınırla
corn_iowa = corn_only.clip(iowa)

# 6. Haritaya eklemek istersen
url = corn_iowa.getThumbURL({
    'min': 1,
    'max': 1,
    'palette': ['yellow'],
    'region': iowa.geometry(),
    'dimensions': 1024
})

print("Corn Mask Visualization URL:", url)
