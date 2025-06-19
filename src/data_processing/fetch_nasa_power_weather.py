import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import logging
import os
import csv

# Set up logging with current time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Add date format
    handlers=[
        logging.FileHandler('data/climate/weather_fetch.log'),
        logging.StreamHandler()
    ]
)

# Read county names and years from usda_yield.csv
def read_counties_and_years_states():
    df = pd.read_csv('data/yield/usda_yield.csv')
    # Only keep valid counties and years
    valid_years = ['2018','2019','2020','2021','2022']
    df = df[df['Year'].astype(str).isin(valid_years)]
    df = df[~df['County'].str.upper().isin(['OTHER (COMBINED) COUNTIES', 'OTHER COUNTIES'])]
    # Drop duplicates for (State, County, Year)
    df = df[['State', 'County', 'Year']].drop_duplicates()
    # Uppercase county for matching
    df['County'] = df['County'].str.upper()
    return df

# County coordinates (latitude, longitude)
COUNTY_COORDS = {
    'ADAIR': (41.3308, -94.4711),
    'ADAMS': (41.0290, -94.6992),
    'ALLAMAKEE': (43.2847, -91.3789),
    'APPANOOSE': (40.7432, -92.8686),
    'AUDUBON': (41.6845, -94.9058),
    'BENTON': (42.0801, -92.0653),
    'BLACK HAWK': (42.4700, -92.3089),
    'BOONE': (42.0333, -93.9333),
    'BREMER': (42.7767, -92.3183),
    'BUCHANAN': (42.4700, -91.8383),
    'BUENA VISTA': (42.7353, -95.1514),
    'BUTLER': (42.7317, -92.7917),
    'CALHOUN': (42.3850, -94.6400),
    'CARROLL': (42.0683, -94.8667),
    'CASS': (41.3317, -94.9283),
    'CEDAR': (41.7717, -91.1317),
    'CERRO GORDO': (43.0817, -93.2600),
    'CHEROKEE': (42.7317, -95.6233),
    'CHICKASAW': (43.0600, -92.3183),
    'CLARKE': (41.0283, -93.7850),
    'CLAY': (43.0817, -95.1514),
    'CLAYTON': (42.8433, -91.3417),
    'CLINTON': (41.8983, -90.5317),
    'CRAWFORD': (42.0350, -95.3817),
    'DALLAS': (41.6850, -94.0400),
    'DAVIS': (40.7475, -92.4117),
    'DECATUR': (40.7375, -93.7867),
    'DELAWARE': (42.4717, -91.3667),
    'DES MOINES': (40.9217, -91.1817),
    'DICKINSON': (43.3783, -95.1514),
    'DUBUQUE': (42.4683, -90.8817),
    'EMMET': (43.3783, -94.6783),
    'FAYETTE': (42.8683, -91.8417),
    'FLOYD': (43.0600, -92.7883),
    'FRANKLIN': (42.7317, -93.2600),
    'FREMONT': (40.7450, -95.6233),
    'GREENE': (42.0350, -94.3967),
    'GRUNDY': (42.4017, -92.7883),
    'GUTHRIE': (41.6833, -94.5000),
    'HAMILTON': (42.3833, -93.7067),
    'HANCOCK': (43.0817, -93.7333),
    'HARDIN': (42.3833, -93.2400),
    'HARRISON': (41.6833, -95.8167),
    'HENRY': (40.9883, -91.5417),
    'HOWARD': (43.3567, -92.3183),
    'HUMBOLDT': (42.7767, -94.2067),
    'IDA': (42.3867, -95.5133),
    'IOWA': (41.6867, -92.0653),
    'JACKSON': (42.1717, -90.5733),
    'JASPER': (41.6867, -93.0533),
    'JEFFERSON': (41.0317, -91.9483),
    'JOHNSON': (41.6717, -91.5883),
    'JONES': (42.1217, -91.1317),
    'KEOKUK': (41.3367, -92.1817),
    'KOSSUTH': (43.2042, -94.2067),
    'LEE': (40.6417, -91.4783),
    'LINN': (42.0783, -91.5983),
    'LOUISA': (41.2183, -91.2600),
    'LUCAS': (41.0283, -93.3283),
    'LYON': (43.3783, -96.2067),
    'MADISON': (41.3308, -94.0150),
    'MAHASKA': (41.3350, -92.6400),
    'MARION': (41.3342, -93.1000),
    'MARSHALL': (42.0350, -92.9983),
    'MILLS': (41.0333, -95.6233),
    'MITCHELL': (43.3567, -92.7883),
    'MONONA': (42.0517, -95.9583),
    'MONROE': (41.0283, -92.8686),
    'MONTGOMERY': (41.0300, -95.1514),
    'MUSCATINE': (41.4817, -91.1133),
    'O BRIEN': (43.0817, -95.6233),
    'OSCEOLA': (43.3783, -95.6233),
    'PAGE': (40.7392, -95.1514),
    'PALO ALTO': (43.0817, -94.6783),
    'PLYMOUTH': (42.7353, -96.2133),
    'POCAHONTAS': (42.7317, -94.6783),
    'POLK': (41.6867, -93.5733),
    'POTTAWATTAMIE': (41.3367, -95.5417),
    'POWESHIEK': (41.6867, -92.5317),
    'RINGGOLD': (40.7350, -94.2433),
    'SAC': (42.3867, -95.1050),
    'SCOTT': (41.6367, -90.6233),
    'SHELBY': (41.6850, -95.3100),
    'SIOUX': (43.0817, -96.1783),
    'STORY': (42.0350, -93.4650),
    'TAMA': (42.0817, -92.5317),
    'TAYLOR': (40.7375, -94.6967),
    'UNION': (41.0283, -94.2433),
    'VAN BUREN': (40.7533, -91.9500),
    'WAPELLO': (41.0333, -92.4100),
    'WARREN': (41.3350, -93.5600),
    'WASHINGTON': (41.3350, -91.7183),
    'WAYNE': (40.7392, -93.3283),
    'WEBSTER': (42.4283, -94.1817),
    'WINNEBAGO': (43.3783, -93.7333),
    'WINNESHIEK': (43.2900, -91.8417),
    'WOODBURY': (42.3900, -96.0450),
    'WORTH': (43.3783, -93.2600),
    'WRIGHT': (42.7317, -93.7333)
}

def write_weather_data_to_csv(writer, daily_data):
    """
    Write weather data directly to CSV file
    """
    if daily_data:
        for data in daily_data:
            writer.writerow(data)

def fetch_single_weather_data(lat, lon, start_date, end_date, state, county):
    """NASA POWER API'den tek bir lokasyon için hava durumu verilerini çeker"""
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Tarihleri YYYYMMDD formatına çevir
    start_date_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end_date_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,WS2M,RH2M,ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": str(lon),
        "latitude": str(lat),
        "start": start_date_fmt,
        "end": end_date_fmt,
        "format": "JSON",
        "header": "true",
        "time-standard": "UTC"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Veri kontrolü
        if 'properties' not in data or 'parameter' not in data['properties']:
            logging.error(f"API yanıtında beklenen veri yapısı bulunamadı: {state} {county}")
            return None
            
        daily_data = []
        parameters = data['properties']['parameter']
        
        # Tarih listesi oluştur
        dates = pd.date_range(start=start_date, end=end_date)
        
        for date in dates:
            date_str = date.strftime('%Y%m%d')
            try:
                daily_values = {
                    'date': date.strftime('%Y-%m-%d'),
                    'state': state,
                    'county': county,
                    'temp_avg': parameters['T2M'][date_str],
                    'temp_max': parameters['T2M_MAX'][date_str],
                    'temp_min': parameters['T2M_MIN'][date_str],
                    'precipitation': parameters['PRECTOTCORR'][date_str],
                    'wind_speed': parameters['WS2M'][date_str],
                    'humidity': parameters['RH2M'][date_str],
                    'solar_radiation': parameters['ALLSKY_SFC_SW_DWN'][date_str]
                }
                daily_data.append(daily_values)
            except KeyError as e:
                logging.error(f"Tarih için veri bulunamadı: {date_str} - {state} {county} - Hata: {str(e)}")
                continue
                
        return daily_data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API hatası - {state} {county}: {str(e)}")
        if 'response' in locals():
            logging.error(f"URL: {response.url}")
            logging.error(f"Response: {response.text}")
        return None
    except Exception as e:
        logging.error(f"Beklenmeyen hata - {state} {county}: {str(e)}")
        return None

def main():
    # Çıktı dizinini oluştur
    os.makedirs('data/climate', exist_ok=True)
    
    # CSV dosyasını oluştur
    output_file = 'data/climate/weather_data_all.csv'
    fieldnames = ['date', 'state', 'county', 'temp_avg', 'temp_max', 'temp_min', 
                 'precipitation', 'wind_speed', 'humidity', 'solar_radiation']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        total_points = len(COUNTY_COORDS) * 6  # 6 yıl için
        processed_points = 0
        successful_points = 0
        
        # 2018-2023 yılları için veri çek
        for year in range(2018, 2024):
            logging.info(f"\n{year} yılı için veri çekiliyor...")
            
            for county, (lat, lon) in COUNTY_COORDS.items():
                processed_points += 1
                logging.info(f"İşleniyor: {county} ({processed_points}/{total_points})")
                
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                daily_data = fetch_single_weather_data(lat, lon, start_date, end_date, "IOWA", county)
                
                if daily_data:
                    write_weather_data_to_csv(writer, daily_data)
                    successful_points += 1
                    logging.info(f"Başarılı: {county} - {year}")
                else:
                    logging.error(f"Başarısız: {county} - {year}")
                
                # API rate limit için bekleme
                time.sleep(1)
            
            logging.info(f"{year} yılı tamamlandı. Başarı oranı: {successful_points}/{processed_points}")
        
        logging.info(f"\nTüm veri çekme işlemi tamamlandı.")
        logging.info(f"Toplam başarı oranı: {successful_points}/{total_points}")

if __name__ == "__main__":
    main() 