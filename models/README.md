Models: 

--- model_rf_1 (Baseline) ---

    Accuracy → 0.6374

   RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        random_state=42
)

Features: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, MONTH, DAY, DAY_OF_WEEK
DISTANCE, DEP_HOUR

Sampling: Random sample: 50k

Class weight: None

Accuracy: 0.6374

Yorum: Model çoğunluk sınıfını (DELAYED=0) tahmin ederek yüksek accuracy elde etti.
Gecikmeli uçuşları yakalama yeteneği düşüktü.



--- model_rf_2 (Feature Engineering + Class Balance) ---

    Class Distribution:
    0 → 63.5%
    1 → 36.5%

    Yeni Features:
        IS_PEAK_HOUR (delay oranına göre öğrenildi)
        IS_WEEKEND
        DISTANCE_BIN

    RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        class_weight="balanced",
        random_state=42
)

    Model Accuracy: 0.6002

    Yorum: Accuracy düştü ancak sınıf dengesizliği ele alındı.Model artık gecikmeli uçuşları tahmin etmeye çalışıyor.
    Bu sonuç gerçek dünya performansı açısından daha anlamlıdır.