"""
Optimized tarımsal veri hazırlama modülü

Bu modül başarılı yield prediction research'lerinden öğrenilen 
en etkili teknikleri uygular:

1. Core vegetation indices (NDVI, EVI, SAVI, NDRE)
2. Water stress index (evapotranspiration based) 
3. Growing degree days (GDD)
4. Seasonal segmentation (3 growth stages)
5. Enhanced temperature features
6. Historical comparison features

Veri akışı:
Raw Data -> County Mapping -> Train/Test Split -> Core Features -> 
Advanced Agricultural Metrics -> Seasonal Features -> Final Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_county_mapping():
    """
    County mapping oluşturur
    
    Returns:
        dict: county_to_id mapping dictionary
    """
    print("County mapping oluşturuluyor...")
    
    df = pd.read_csv('data/satellite/processed/test_sentinel.csv')
    unique_counties = df['county_name'].unique()
    unique_counties = [county for county in unique_counties if pd.notna(county)]
    unique_counties_lower = list(set([str(county).lower().strip() for county in unique_counties]))
    unique_counties_lower.sort()
    
    county_to_id = {county: idx for idx, county in enumerate(unique_counties_lower)}
    
    print(f"Toplam {len(county_to_id)} farklı il bulundu")
    return county_to_id

def month_to_number(month):
    """Ay ismini sayıya çevirir"""
    month_map = {
        'ocak': 1, 'şubat': 2, 'subat': 2, 'mart': 3, 'nisan': 4,
        'mayıs': 5, 'mayis': 5, 'haziran': 6, 'temmuz': 7,
        'ağustos': 8, 'agustos': 8, 'eylül': 9, 'eylul': 9,
        'ekim': 10, 'kasım': 11, 'kasim': 11, 'aralık': 12, 'aralik': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    if pd.isna(month):
        return np.nan
        
    m = str(month).lower().strip()
    return month_map.get(m, np.nan)

def calculate_core_vegetation_indices(df):
    """
    Research-proven core vegetation indices hesaplar
    
    Prince Edward Island ve diğer başarılı çalışmalarda
    en etkili bulunan 4 temel indeks:
    - NDVI: Genel bitki sağlığı (en yaygın ve güvenilir)
    - EVI: Atmosferik düzeltmeli, yoğun bitki örtüsü için
    - SAVI: Toprak etkilerini azaltır, seyrek bitki için
    - NDRE: Red edge, chlorophyll içeriği için
    
    Gereksiz indeksler çıkarıldı: CIG, GNDVI, MCARI, MSR
    """
    print("Core vegetation indeksleri hesaplanıyor...")
    
    df_result = df.copy()
    
    def safe_divide(numerator, denominator, default_value=np.nan):
        return np.where(denominator != 0, numerator / denominator, default_value)
    
    # NDVI - En temel ve güvenilir indeks
    numerator = df['B8'] - df['B4']
    denominator = df['B8'] + df['B4']
    df_result['NDVI'] = safe_divide(numerator, denominator)
    
    # EVI - Atmosferik düzeltmeli, research'lerde çok başarılı
    numerator = 2.5 * (df['B8'] - df['B4'])
    denominator = df['B8'] + 6*df['B4'] - 7.5*df['B2'] + 1
    df_result['EVI'] = safe_divide(numerator, denominator)
    
    # SAVI - Toprak düzeltmeli, PEI çalışmasında etkili
    numerator = (df['B8'] - df['B4']) * 1.5
    denominator = df['B8'] + df['B4'] + 0.5
    df_result['SAVI'] = safe_divide(numerator, denominator)
    
    # NDRE - Red edge, chlorophyll için kritik
    numerator = df['B8'] - df['B5']
    denominator = df['B8'] + df['B5']
    df_result['NDRE'] = safe_divide(numerator, denominator)
    
    # İstatistikler
    indices = ['NDVI', 'EVI', 'SAVI', 'NDRE']
    print(f"Hesaplanan core vegetation indeksleri ({len(indices)} adet):")
    for idx in indices:
        valid_count = df_result[idx].notna().sum()
        mean_val = df_result[idx].mean()
        print(f"  {idx}: {valid_count} geçerli, ortalama: {mean_val:.4f}")
    
    return df_result

def calculate_growing_degree_days(temp_avg, base_temp=10):
    """
    Growing Degree Days (GDD) hesaplar
    
    Prince Edward Island çalışmasında en önemli feature'lardan biri.
    Bitki gelişimi için kritik thermal index.
    
    Args:
        temp_avg: Ortalama sıcaklık
        base_temp: Baz sıcaklık (corn için genellikle 10°C)
    
    Returns:
        GDD değeri (pozitif değerler, bitki büyümesi için)
    """
    return np.maximum(temp_avg - base_temp, 0)

def calculate_water_stress_index(temp_avg, humidity, precipitation):
    """
    Water Stress Index hesaplar
    
    Prince Edward Island çalışmasında top predictor.
    Simplified version of evapotranspiration-based index.
    
    Formula: Basitleştirilmiş ET0 estimation
    """
    # Simplified ET0 (Reference evapotranspiration)
    # Thornthwaite method approximation
    et0_approx = np.maximum(0, 0.016 * temp_avg * np.sqrt(np.maximum(temp_avg, 0)))
    
    # Water availability (humidity + precipitation effect)
    water_availability = (humidity / 100) + np.sqrt(np.maximum(precipitation, 0)) / 10
    
    # Water stress: higher values = more stress
    water_stress = et0_approx / np.maximum(water_availability, 0.1)
    
    return water_stress

def get_growth_stage(month_num):
    """
    Tarımsal büyüme evresini belirler
    
    Prince Edward Island approach: 3 dönem
    - Early (4-6): Ekim, fide, erken gelişim
    - Mid (7-9): Büyüme, çiçeklenme, dolum
    - Late (10-11): Olgunlaşma, hasat öncesi
    """
    if pd.isna(month_num):
        return 'unknown'
    
    if month_num in [4, 5, 6]:
        return 'early'
    elif month_num in [7, 8, 9]:
        return 'mid'
    elif month_num in [10, 11]:
        return 'late'
    else:
        return 'off_season'

def calculate_enhanced_weather_features(df_weather):
    """
    Enhanced weather features hesaplar
    
    Research-proven agricultural weather metrics:
    - GDD (Growing Degree Days)
    - Water stress index
    - Temperature stress indicators
    - Growth stage specific features
    """
    print("Enhanced weather features hesaplanıyor...")
    
    df_enhanced = df_weather.copy()
    
    # Growing Degree Days (base 10°C for corn)
    df_enhanced['gdd'] = calculate_growing_degree_days(df_enhanced['temp_avg'], base_temp=10)
    
    # Water stress index
    df_enhanced['water_stress'] = calculate_water_stress_index(
        df_enhanced['temp_avg'], 
        df_enhanced['humidity'], 
        df_enhanced['precipitation']
    )
    
    # Temperature stress indicators
    df_enhanced['heat_stress'] = np.maximum(df_enhanced['temp_max'] - 30, 0)  # Heat stress above 30°C
    df_enhanced['cold_stress'] = np.maximum(5 - df_enhanced['temp_min'], 0)  # Cold stress below 5°C
    
    # Diurnal temperature range (important for crop quality)
    df_enhanced['temp_range'] = df_enhanced['temp_max'] - df_enhanced['temp_min']
    
    # Growth stage assignment
    df_enhanced['growth_stage'] = df_enhanced['month_num'].apply(get_growth_stage)
    
    print("Enhanced weather features:")
    new_features = ['gdd', 'water_stress', 'heat_stress', 'cold_stress', 'temp_range']
    for feature in new_features:
        mean_val = df_enhanced[feature].mean()
        print(f"  {feature}: ortalama {mean_val:.4f}")
    
    return df_enhanced

def split_and_write_sentinel_data(county_mapping):
    """
    Optimized sentinel data processing
    
    Sadece core vegetation indices ile daha clean dataset
    """
    print(f"\nSentinel verileri optimize edilerek işleniyor...")
    
    df = pd.read_csv('data/satellite/processed/test_sentinel.csv')
    
    # County processing
    df['county_name_lower'] = df['county_name'].str.lower().str.strip()
    df['county_id'] = df['county_name_lower'].map(county_mapping)
    df['month_num'] = df['month'].apply(month_to_number)
    
    # Core sentinel bands (B10 excluded)
    band_columns = [f'B{i}' for i in range(1, 13) if i != 10] + ['B8A']
    
    # Select essential columns
    output_columns = ['county_id', 'year', 'month_num', 'date'] + band_columns
    df_processed = df[output_columns].copy()
    
    # Calculate only core vegetation indices
    df_processed = calculate_core_vegetation_indices(df_processed)
    
    # Temporal split
    df_test = df_processed[df_processed['year'] == 2023].copy()
    df_train = df_processed[df_processed['year'] != 2023].copy()
    
    # Shuffle
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Write files
    df_train.to_csv('data/processed/train_data.csv', index=False)
    df_test.to_csv('data/processed/test_data.csv', index=False)
    
    print(f"Optimized train data: {len(df_train)} kayıt")
    print(f"Optimized test data: {len(df_test)} kayıt")
    print("Features: county_id, year, month_num, date, core bands, 4 vegetation indices")
    
def append_enhanced_weather_to_data(data_type='train'):
    """
    Enhanced weather data with agricultural features
    """
    print(f"\n{data_type.title()} datasına enhanced weather ekleniyor...")
    
    df_data = pd.read_csv(f'data/processed/{data_type}_data.csv')
    df_weather = pd.read_csv('data/climate/weather_data_all_test.csv')
    
    # Date processing
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather['year'] = df_weather['date'].dt.year
    df_weather['month_num'] = df_weather['date'].dt.month
    
    if 'county' in df_weather.columns:
        df_weather['county_name_lower'] = df_weather['county'].str.lower().str.strip()
    
    # Enhanced daily weather features
    df_weather_enhanced = calculate_enhanced_weather_features(df_weather)
    
    # Monthly aggregation with growth-stage awareness
    weather_monthly = df_weather_enhanced.groupby(
        ['county_name_lower', 'year', 'month_num'], as_index=False
    ).agg({
        'temp_avg': 'mean',
        'temp_max': 'max',
        'temp_min': 'min',
        'precipitation': 'sum',
        'humidity': 'mean',
        'solar_radiation': 'mean',
        # Enhanced features
        'gdd': 'sum',                    # Cumulative GDD per month
        'water_stress': 'mean',          # Average water stress
        'heat_stress': 'sum',            # Cumulative heat stress
        'cold_stress': 'sum',            # Cumulative cold stress
        'temp_range': 'mean'             # Average diurnal temperature range
    })
    
    # Add county mapping for merge
    if 'county_name_lower' not in df_data.columns:
        county_mapping = create_county_mapping()
        id_to_county = {v: k for k, v in county_mapping.items()}
        df_data['county_name_lower'] = df_data['county_id'].map(id_to_county)
    
    # Merge enhanced weather
    df_final = df_data.merge(
        weather_monthly,
        on=['county_name_lower', 'year', 'month_num'], 
        how='left'
    )
    
    # Add growth stage info
    df_final['growth_stage'] = df_final['month_num'].apply(get_growth_stage)
    
    # Cleanup
    df_final = df_final.drop('county_name_lower', axis=1)
    
    # Write updated file
    df_final.to_csv(f'data/processed/{data_type}_data.csv', index=False)
    
    print(f"Enhanced weather eklendi: {df_final.shape[0]} kayıt, {df_final.shape[1]} sütun")
    print("Yeni features: GDD, water_stress, heat_stress, cold_stress, temp_range, growth_stage")

def clean_numeric_value(value):
    """Numeric value cleaning"""
    if pd.isna(value):
        return np.nan
    
    value_str = str(value)
    
    if value_str.strip() == '' or value_str.strip().lower() in ['nan', 'null', 'none']:
        return np.nan
    
    import re
    cleaned = re.sub(r'[^\d.,]', '', value_str)
    
    if cleaned == '':
        return np.nan
    
    cleaned = cleaned.replace(',', '.')
    
    if cleaned.count('.') > 1:
        parts = cleaned.split('.')
        if len(parts) > 1:
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
    
    try:
        return float(cleaned)
    except ValueError:
        print(f"Değer temizlenemedi: '{value}' -> '{cleaned}'")
        return np.nan

def append_usda_yield_to_data(county_mapping, data_type):
    """USDA yield data addition - unchanged logic"""
    print(f"\n{data_type.title()} datasına USDA yield verisi ekleniyor...")
    
    data_df = pd.read_csv(f'data/processed/{data_type}_data.csv')
    print(f"Orijinal veri boyutu: {data_df.shape[0]} kayıt")
    
    yield_file_paths = [
        'data/yield/usda_yield.csv',
        'data/yield/usda_yield_test.csv',
        'usda_yield.csv'
    ]
    
    df_yield = None
    for file_path in yield_file_paths:
        try:
            df_yield = pd.read_csv(file_path)
            print(f"USDA yield dosyası bulundu: {file_path}")
            break
        except FileNotFoundError:
            continue
    
    if df_yield is None:
        print("UYARI: USDA yield dosyası bulunamadı!")
        return data_df
    
    # Filter corn yield data
    corn_items = ['CORN, GRAIN - YIELD, MEASURED IN BU / ACRE']
    df_yield_filtered = None
    
    for item in corn_items:
        if 'Data Item' in df_yield.columns:
            df_temp = df_yield[df_yield['Data Item'] == item]
            if len(df_temp) > 0:
                df_yield_filtered = df_temp.copy()
                print(f"Corn yield verisi bulundu: {len(df_yield_filtered)} kayıt")
                break
    
    if df_yield_filtered is None:
        df_yield_filtered = df_yield.copy()
        print(f"Corn filtrelemesi yapılamadı, tüm veri kullanılıyor")
    
    # Find column names dynamically
    county_column = None
    for col in ['County', 'county', 'County Name', 'county_name']:
        if col in df_yield_filtered.columns:
            county_column = col
            break
    
    year_column = None
    for col in ['Year', 'year', 'YEAR']:
        if col in df_yield_filtered.columns:
            year_column = col
            break
    
    value_column = None
    for col in ['Value', 'value', 'yield', 'Yield', 'VALUE']:
        if col in df_yield_filtered.columns:
            value_column = col
            break
    
    if not all([county_column, year_column, value_column]):
        print("UYARI: Gerekli sütunlar bulunamadı!")
        return data_df
    
    # Process yield data
    df_yield_filtered['county_name_lower'] = df_yield_filtered[county_column].str.lower().str.strip()
    df_yield_filtered['county_id'] = df_yield_filtered['county_name_lower'].map(county_mapping)
    df_yield_filtered['yield'] = df_yield_filtered[value_column].apply(clean_numeric_value)
    
    # Prepare for merge
    yield_data = df_yield_filtered[['county_id', year_column, 'yield']].rename(
        columns={year_column: 'year'}
    )
    yield_data = yield_data.dropna(subset=['county_id', 'year', 'yield'])
    
    # Group by county and year
    yield_data_grouped = yield_data.groupby(['county_id', 'year'], as_index=False).agg({
        'yield': 'mean'
    })
    
    # Merge with main data
    data_merged = data_df.merge(
        yield_data_grouped, 
        on=['county_id', 'year'], 
        how='left'
    )
    
    # Save
    data_merged.to_csv(f'data/processed/{data_type}_data.csv', index=False)
    
    yield_count = data_merged['yield'].notna().sum()
    total_count = len(data_merged)
    print(f"Yield merge sonucu: {yield_count}/{total_count} ({yield_count/total_count*100:.1f}%)")
    
    return data_merged

def create_seasonal_features(data_type='train'):
    """
    Seasonal aggregation features ekler
    
    Prince Edward Island approach: Growth stage bazında feature'lar
    Bu model performance'ı önemli ölçüde iyileştirebilir.
    """
    print(f"\n{data_type.title()} datasına seasonal features ekleniyor...")
    
    df = pd.read_csv(f'data/processed/{data_type}_data.csv')
    
    # Growth stage bazında aggregation
    seasonal_features = []
    
    for stage in ['early', 'mid', 'late']:
        stage_data = df[df['growth_stage'] == stage]
        
        if len(stage_data) > 0:
            stage_agg = stage_data.groupby(['county_id', 'year'], as_index=False).agg({
                'NDVI': 'mean',
                'EVI': 'mean', 
                'SAVI': 'mean',
                'NDRE': 'mean',
                'temp_avg': 'mean',
                'precipitation': 'sum',
                'gdd': 'sum',
                'water_stress': 'mean'
            })
            
            # Rename columns with stage prefix
            for col in ['NDVI', 'EVI', 'SAVI', 'NDRE', 'temp_avg', 'precipitation', 'gdd', 'water_stress']:
                stage_agg = stage_agg.rename(columns={col: f'{stage}_{col}'})
            
            seasonal_features.append(stage_agg)
    
    # Merge all seasonal features
    if seasonal_features:
        seasonal_df = seasonal_features[0]
        for stage_df in seasonal_features[1:]:
            seasonal_df = seasonal_df.merge(stage_df, on=['county_id', 'year'], how='outer')
        
        # Merge with main data
        df_with_seasonal = df.merge(seasonal_df, on=['county_id', 'year'], how='left')
        
        # Save
        df_with_seasonal.to_csv(f'data/processed/{data_type}_data.csv', index=False)
        
        print(f"Seasonal features eklendi: {df_with_seasonal.shape[1] - df.shape[1]} yeni sütun")
        print("Seasonal stages: early (4-6), mid (7-9), late (10-11)")
    else:
        print("Seasonal feature oluşturulamadı")

if __name__ == "__main__":
    """
    Optimized tarımsal veri hazırlama pipeline'ı
    
    Research-proven techniques:
    ✅ Core vegetation indices (4 instead of 8)
    ✅ Growing degree days (GDD)
    ✅ Water stress index
    ✅ Enhanced temperature features
    ✅ Seasonal segmentation
    ✅ Growth stage features
    """
    print("=== OPTIMIZED Tarımsal Veri Hazırlama Pipeline'ı ===")
    print("Research-proven techniques applied:")
    print("✅ Core vegetation indices (NDVI, EVI, SAVI, NDRE)")
    print("✅ Growing degree days (GDD)")
    print("✅ Water stress index")
    print("✅ Enhanced temperature features")
    print("✅ Seasonal segmentation")
    
    # 1. County mapping
    print("\n1. County mapping oluşturuluyor...")
    county_mapping = create_county_mapping()
    
    # 2. Optimized satellite processing
    print("\n2. Satellite verileri optimize edilerek işleniyor...")
    split_and_write_sentinel_data(county_mapping)
    
    # 3. Enhanced weather features
    print("\n3. Enhanced weather features ekleniyor...")
    append_enhanced_weather_to_data('train')
    append_enhanced_weather_to_data('test')
    
    # 4. USDA yield data
    print("\n4. USDA yield verileri ekleniyor...")
    append_usda_yield_to_data(county_mapping, 'train')
    append_usda_yield_to_data(county_mapping, 'test')
    
    # 5. Seasonal features (NEW!)
    print("\n5. Seasonal features oluşturuluyor...")
    create_seasonal_features('train')
    create_seasonal_features('test')
    
    print("\n=== OPTIMIZATION COMPLETE ===")
    print("\nKey improvements:")
    print("📊 Vegetation indices: 8 → 4 (core research-proven)")
    print("🌡️  Temperature features: Enhanced with GDD, stress indicators")
    print("💧 Water features: Added water stress index")
    print("📅 Temporal features: Growth stage segmentation")
    print("🎯 Target: Optimized for yield prediction performance")
    print("\nReady for advanced ML training!")
