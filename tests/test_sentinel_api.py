import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import random
import json

def get_iowa_county():
    """
    Iowa county sınırlarını içeren örnek bir GeoJSON dosyası oluşturur ve okur.
    """
    counties_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"NAME": "Polk", "COUNTYFP": "153"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-93.7, 41.6], [-93.6, 41.6], [-93.6, 41.7], [-93.7, 41.7], [-93.7, 41.6]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME": "Story", "COUNTYFP": "169"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-93.7, 42.0], [-93.6, 42.0], [-93.6, 42.1], [-93.7, 42.1], [-93.7, 42.0]]]
                }
            }
        ]
    }
    os.makedirs("data/field", exist_ok=True)
    with open("data/field/iowa_counties.geojson", "w") as f:
        json.dump(counties_data, f)
    counties = gpd.read_file("data/field/iowa_counties.geojson")
    return counties

def main():
    # GEE'yi başlat
    ee.Initialize(project='agr-46')

    counties = get_iowa_county()
    random_county = counties.sample(n=1).iloc[0]
    county_name = random_county["NAME"]
    county_geom = random_county["geometry"]
    print(f"Seçilen county: {county_name}")

    # GeoPandas geometrisini GEE geometry'ye çevir
    coords = county_geom.__geo_interface__["coordinates"]
    gee_geom = ee.Geometry.Polygon(coords)

    # Sentinel-2 koleksiyonunu al
    collection = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(gee_geom) \
        .filterDate('2023-05-01', '2023-06-01') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    image = collection.median()

    bands = {
        "B1": "B01",  # Coastal aerosol
        "B2": "B02",  # Blue
        "B3": "B03",  # Green
        "B4": "B04",  # Red
        "B5": "B05",  # Vegetation red edge
        "B6": "B06",  # Vegetation red edge
        "B7": "B07",  # Vegetation red edge
        "B8": "B08",  # NIR
        "B8A": "B8A", # Narrow NIR
        "B9": "B09",  # Water vapour
        "B11": "B11", # SWIR1
        "B12": "B12"  # SWIR2
    }

    results = []
    for gee_band, label in bands.items():
        mean_dict = image.select(gee_band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=gee_geom,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        mean_val = mean_dict.get(gee_band, None)
        print(f"{label} bandı ortalama değeri: {mean_val}")
        results.append({"band": label, "mean_value": mean_val})

    df = pd.DataFrame(results)
    os.makedirs("data/satellite", exist_ok=True)
    output_file = f"data/satellite/{county_name}_sentinel2_bands_gee.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSonuçlar kaydedildi: {output_file}")

if __name__ == "__main__":
    main() 