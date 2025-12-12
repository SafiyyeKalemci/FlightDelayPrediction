import pandas as pd
import os

# Dosya yolu (her zaman data klasöründen çeker)
DATA_PATH = os.path.join("..", "data", "flights.csv")

# 1. Veri setini yükle
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Original data:", df.shape)

# 2. Tahmin için gerekli sütunları seç
cols_to_use = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DISTANCE",
    "ARRIVAL_DELAY"  # TARGET
]

df = df[cols_to_use]

# 3. Cancelled olan uçuşları çıkar (tahmin edilemez)
df = df[df["ARRIVAL_DELAY"].notna()]

# 4. SCHEDULED_DEPARTURE → 0005 → 0:05 → sadece saat olarak alalım
def convert_to_hour(x):
    """0005 → 0, 1340 → 13 gibi sadece hour"""
    x = int(x)
    hour = x // 100
    if hour < 0 or hour > 23:
        return None
    return hour

df["DEP_HOUR"] = df["SCHEDULED_DEPARTURE"].apply(convert_to_hour)

# SCHEDULED_DEPARTURE artık gereksiz → kaldır
df = df.drop(columns=["SCHEDULED_DEPARTURE"])

# 5. Eksik verileri temizle (boş olan satırları siler)
df = df.dropna()

# 6. Arrival delay'i sınıfa dönüştür:
#    0: zamanında ya da erken; False → 0
#    1: gecikmeli, True → 1
df["DELAYED"] = (df["ARRIVAL_DELAY"] > 0).astype(int)

# ARRIVAL_DELAY’i artık kullanmıyoruz → kaldırabilirsin
# (istersen regression yapacaksan tutabilirsiniz)
df = df.drop(columns=["ARRIVAL_DELAY"])

print("Cleaned data shape:", df.shape)

# 7. Temiz veriyi kaydet.
OUTPUT_PATH = os.path.join("..","data", "clean_flights.csv")
df.to_csv(OUTPUT_PATH, index=False)

print(f"TEMİZ VERİ KAYDEDİLDİ → {OUTPUT_PATH}")