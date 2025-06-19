import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.usda_yield_processor import UsdaYieldProcessor

def test_usda():
    processor = UsdaYieldProcessor()
    county = "Story"
    year = 2021
    value = processor.fetch_usda_corn_data(county, year)
    print(f"{county} ({year}) için USDA verimi: {value}")

if __name__ == "__main__":
    test_usda() 