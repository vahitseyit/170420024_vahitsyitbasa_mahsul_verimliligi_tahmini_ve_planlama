from src.data_processing.climate_data_processor import ClimateDataProcessor

def test_climate():
    processor = ClimateDataProcessor()
    # Örnek: 2021 yılı için veri çek
    try:
        result = processor.collect_and_process(start_year=2021, end_year=2021)
        print("2021 yılı için iklim verisi başarıyla işlendi.")
        print(result)
    except Exception as e:
        print(f"İklim verisi alınırken hata oluştu: {e}")

if __name__ == "__main__":
    test_climate() 