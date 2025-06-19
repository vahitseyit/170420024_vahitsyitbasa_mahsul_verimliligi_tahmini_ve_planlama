import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
USDA_API_KEY = os.getenv('USDA_API_KEY')
USDA_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

class UsdaYieldProcessor:
    def __init__(self):
        pass

    def fetch_usda_corn_data(self, county_name, year):
        state_code = "IA"  # Varsayılan olarak Iowa
        # Eğer ilçe Iowa'da değilse, eyalet kodunu belirle (örnekler)
        if county_name == "Jo Daviess":
            state_code = "IL"
        elif county_name in ["Rock", "Lyon", "Sioux"]:
            state_code = "MN"
        elif county_name in ["Harrison", "Pottawattamie", "Mills", "Fremont", "Page"]:
            state_code = "NE"
        elif county_name in ["Lee", "Des Moines", "Henry"]:
            state_code = "MO"
        elif county_name in ["Lyon", "Sioux"]:
            state_code = "SD"
        elif county_name in ["Allamakee", "Clayton", "Dubuque"]:
            state_code = "WI"
        params = {
            "key": USDA_API_KEY,
            "commodity_desc": "CORN",
            "statisticcat_desc": "PRODUCTION",
            "year": year,
            "format": "JSON",
            "county_name": county_name,
            "state_alpha": state_code,
            "unit_desc": "BU"
        }
        try:
            response = requests.get(USDA_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if not data.get('data'):
                print(f"USDA veri bulunamadı - İlçe: {county_name}, Eyalet: {state_code}, Yıl: {year}")
                return None
            value = data['data'][0].get('Value', None)
            if value is None:
                print(f"USDA değer bulunamadı - İlçe: {county_name}, Eyalet: {state_code}, Yıl: {year}")
                return None
            try:
                value = int(value.replace(',', ''))
                print(f"USDA veri başarıyla alındı - İlçe: {county_name}, Eyalet: {state_code}, Yıl: {year}, Değer: {value}")
                return value
            except Exception as e:
                print(f"USDA değer dönüştürme hatası - İlçe: {county_name}, Eyalet: {state_code}, Yıl: {year}, Hata: {str(e)}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"USDA API hatası - İlçe: {county_name}, Eyalet: {state_code}, Yıl: {year}, Hata: {str(e)}")
            return None
        except Exception as e:
            print(f"USDA beklenmeyen hata - İlçe: {county_name}, Eyalet: {state_code}, Yıl: {year}, Hata: {str(e)}")
            return None

    def process_county_year(self, county_name, year):
        """
        Bir ilçe ve yıl için USDA verisini çekip döndürür.
        """
        return self.fetch_usda_corn_data(county_name, year)

    def process_bulk(self, county_year_list, output_path):
        """
        Birden fazla ilçe-yıl için USDA verilerini toplar ve CSV'ye kaydeder.
        """
        results = []
        for county_name, year in county_year_list:
            value = self.process_county_year(county_name, year)
            results.append({
                'county_name': county_name,
                'year': year,
                'usda_corn_production_bu': value
            })
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"USDA verileri kaydedildi: {output_path}")


def main():
    processor = UsdaYieldProcessor()
    # Örnek toplu kullanım:
    # county_year_list = [("Story", 2021), ("Polk", 2021)]
    # processor.process_bulk(county_year_list, "data/yield/usda_yield_sample.csv")
    pass

if __name__ == "__main__":
    main() 