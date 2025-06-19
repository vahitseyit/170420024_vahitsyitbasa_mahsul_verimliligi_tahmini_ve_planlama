import os
import pandas as pd
from datetime import datetime

class USDAProcessor:
    """USDA verilerini işlemek için sınıf"""
    
    def __init__(self):
        """USDA veri işleyici sınıfı"""
        # Dizinler
        self.raw_dir = "data/usda/raw"
        self.processed_dir = "data/usda/processed"
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Dosya yolları
        self.raw_file = os.path.join(self.raw_dir, "usda_corn_data.csv")
        self.yield_file = os.path.join(self.processed_dir, "iowa_corn_yield_2018_2023.csv")
        self.acres_file = os.path.join(self.processed_dir, "iowa_corn_acres_2018_2023.csv")
        
        # İlçe isimlerini düzeltme sözlüğü
        self.county_name_map = {
            "ADAIR": "Adair",
            "ADAMS": "Adams",
            "ALLAMAKEE": "Allamakee",
            "APPANOOSE": "Appanoose",
            "AUDUBON": "Audubon",
            "BENTON": "Benton",
            "BLACK HAWK": "Black Hawk",
            "BOONE": "Boone",
            "BREMER": "Bremer",
            "BUCHANAN": "Buchanan",
            "BUENA VISTA": "Buena Vista",
            "BUTLER": "Butler",
            "CALHOUN": "Calhoun",
            "CARROLL": "Carroll",
            "CASS": "Cass",
            "CEDAR": "Cedar",
            "CERRO GORDO": "Cerro Gordo",
            "CHEROKEE": "Cherokee",
            "CHICKASAW": "Chickasaw",
            "CLARKE": "Clarke",
            "CLAY": "Clay",
            "CLAYTON": "Clayton",
            "CLINTON": "Clinton",
            "CRAWFORD": "Crawford",
            "DALLAS": "Dallas",
            "DAVIS": "Davis",
            "DECATUR": "Decatur",
            "DELAWARE": "Delaware",
            "DES MOINES": "Des Moines",
            "DICKINSON": "Dickinson",
            "DUBUQUE": "Dubuque",
            "EMMET": "Emmet",
            "FAYETTE": "Fayette",
            "FLOYD": "Floyd",
            "FRANKLIN": "Franklin",
            "FREMONT": "Fremont",
            "GREENE": "Greene",
            "GRUNDY": "Grundy",
            "GUTHRIE": "Guthrie",
            "HAMILTON": "Hamilton",
            "HANCOCK": "Hancock",
            "HARDIN": "Hardin",
            "HARRISON": "Harrison",
            "HENRY": "Henry",
            "HOWARD": "Howard",
            "HUMBOLDT": "Humboldt",
            "IDA": "Ida",
            "IOWA": "Iowa",
            "JACKSON": "Jackson",
            "JASPER": "Jasper",
            "JEFFERSON": "Jefferson",
            "JOHNSON": "Johnson",
            "JONES": "Jones",
            "KEOKUK": "Keokuk",
            "KOSSUTH": "Kossuth",
            "LEE": "Lee",
            "LINN": "Linn",
            "LOUISA": "Louisa",
            "LUCAS": "Lucas",
            "LYON": "Lyon",
            "MADISON": "Madison",
            "MAHASKA": "Mahaska",
            "MARION": "Marion",
            "MARSHALL": "Marshall",
            "MILLS": "Mills",
            "MITCHELL": "Mitchell",
            "MONONA": "Monona",
            "MONROE": "Monroe",
            "MONTGOMERY": "Montgomery",
            "MUSCATINE": "Muscatine",
            "OBRIEN": "O'Brien",
            "OSCEOLA": "Osceola",
            "PAGE": "Page",
            "PALO ALTO": "Palo Alto",
            "PLYMOUTH": "Plymouth",
            "POCAHONTAS": "Pocahontas",
            "POLK": "Polk",
            "POTTAWATTAMIE": "Pottawattamie",
            "POWESHIEK": "Poweshiek",
            "RINGGOLD": "Ringgold",
            "SAC": "Sac",
            "SCOTT": "Scott",
            "SHELBY": "Shelby",
            "SIOUX": "Sioux",
            "STORY": "Story",
            "TAMA": "Tama",
            "TAYLOR": "Taylor",
            "UNION": "Union",
            "VAN BUREN": "Van Buren",
            "WAPELLO": "Wapello",
            "WARREN": "Warren",
            "WASHINGTON": "Washington",
            "WAYNE": "Wayne",
            "WEBSTER": "Webster",
            "WINNEBAGO": "Winnebago",
            "WINNESHIEK": "Winneshiek",
            "WOODBURY": "Woodbury",
            "WORTH": "Worth",
            "WRIGHT": "Wright"
        }
    
    def process_data(self):
        """CSV dosyasını işler ve verileri ayırır"""
        print("USDA verileri işleniyor...")
        
        # CSV dosyasını oku
        if not os.path.exists(self.raw_file):
            print(f"Hata: {self.raw_file} bulunamadı!")
            print("Lütfen USDA QuickStats'ten indirdiğiniz CSV dosyasını bu konuma koyun.")
            return
        
        # Veriyi oku
        df = pd.read_csv(self.raw_file)
        
        # Verim verilerini filtrele
        yield_df = df[
            (df['Statistic Category'] == 'YIELD') & 
            (df['Unit'] == 'BU / ACRE')
        ].copy()
        
        # Ekim alanı verilerini filtrele
        acres_df = df[
            (df['Statistic Category'] == 'AREA PLANTED') & 
            (df['Unit'] == 'ACRES')
        ].copy()
        
        # İlçe isimlerini düzelt
        yield_df['County'] = yield_df['County'].apply(self.fix_county_name)
        acres_df['County'] = acres_df['County'].apply(self.fix_county_name)
        
        # Tarih sütunu ekle
        yield_df['date'] = pd.to_datetime(yield_df['Year'].astype(str) + '-12-31')
        acres_df['date'] = pd.to_datetime(acres_df['Year'].astype(str) + '-12-31')
        
        # Sütun isimlerini düzenle
        yield_df = yield_df.rename(columns={
            'County': 'county_name',
            'Year': 'year',
            'Value': 'yield'
        })
        
        acres_df = acres_df.rename(columns={
            'County': 'county_name',
            'Year': 'year',
            'Value': 'acres'
        })
        
        # Gerekli sütunları seç
        yield_df = yield_df[['county_name', 'year', 'date', 'yield']]
        acres_df = acres_df[['county_name', 'year', 'date', 'acres']]
        
        # Verileri kaydet
        yield_df.to_csv(self.yield_file, index=False)
        acres_df.to_csv(self.acres_file, index=False)
        
        print(f"Verim verileri kaydedildi: {self.yield_file}")
        print(f"Ekim alanı verileri kaydedildi: {self.acres_file}")
    
    def fix_county_name(self, name):
        """İlçe ismini düzeltir"""
        return self.county_name_map.get(name.upper(), name.title())

def main():
    """Ana fonksiyon"""
    processor = USDAProcessor()
    processor.process_data()

if __name__ == "__main__":
    main() 