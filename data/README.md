# Veri Seti Yapısı

Bu klasör, mısır verim tahmini için kullanılan tüm veri setlerini içerir.

## Klasör Yapısı

```
data/
├── satellite/              # Uydu görüntüleri ve indeksler
│   ├── sentinel_images/    # Sentinel-2 ham görüntüleri
│   └── indices/           # Bitki indeksleri (NDVI, EVI, vb.)
├── climate/               # İklim verileri
│   ├── temperature.csv    # Sıcaklık verileri
│   ├── precipitation.csv  # Yağış verileri
│   └── humidity.csv      # Nem verileri
├── soil/                 # Toprak verileri
│   ├── soil_moisture.csv # Toprak nemi
│   ├── soil_temp.csv    # Toprak sıcaklığı
│   └── soil_ph.csv      # Toprak pH değeri
├── field/               # Tarla verileri
│   ├── characteristics.csv # Tarla özellikleri
│   └── management.csv    # Yönetim uygulamaları
├── yield/              # Verim verileri
│   └── usda_yield.csv  # USDA verim verileri
└── features/          # Model için hazırlanmış özellikler
```

## Veri Formatları

### Uydu Verileri
- Sentinel-2 görüntüleri: GeoTIFF formatında
- İndeksler: CSV formatında
- Çözünürlük: 10m
- Projeksiyon: UTM

### İklim Verileri
- Günlük/haftalık veriler
- CSV formatında
- Tarih formatı: YYYY-MM-DD

### Toprak Verileri
- Aylık/günlük ölçümler
- CSV formatında
- Koordinat bazlı

### Tarla Verileri
- Yıllık güncelleme
- CSV formatında
- Tarla ID bazlı

### Verim Verileri
- Yıllık veriler
- CSV formatında
- İlçe bazlı

## Veri İşleme

1. Veriler doğrudan ilgili klasörlere yazılır
2. Özellik çıkarılmış veriler `features/` klasöründe saklanır

## Veri Güncelleme

- Uydu verileri: 10 günde bir
- İklim verileri: Günlük
- Toprak verileri: Aylık
- Tarla verileri: Yıllık
- Verim verileri: Yıllık 