import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

PEAK_HOURS = None

# Path Ayarları
DATA_PATH = os.path.join("..", "data", "clean_flights.csv")
MODEL_PATH = os.path.join("..", "models", "flight_delay_model.pkl")

# Veriyi yükler ve DataFrame döner.
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No data found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, low_memory=False)

# Feature/target ayırır ve train/test split yapar.
# Ayrıca preprocessing pipeline’ı da döndürür.
def preprocess_and_split(df):
    X = df.drop(columns=["DELAYED"])
    y = df["DELAYED"]

    # Kategorik & sayısal kolonlar
    categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DISTANCE_BIN"]
    numeric_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "DISTANCE", "DEP_HOUR", "IS_PEAK_HOUR", "IS_WEEKEND" ]

    for col in categorical_cols:
        X[col] = X[col].astype(str)

    # Preprocessing pipeline (One-hot encoding)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

# Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocess

# Modeli eğitir ve models/ altına kaydeder.
def train_model():
    global PEAK_HOURS

    print("Veri yükleniyor...")
    df = load_data()

    print("Class distribution:")
    print(df["DELAYED"].value_counts(normalize=True))

    # FEATURE ENGINEERING
    delay_by_hour = df.groupby("DEP_HOUR")["DELAYED"].mean()
    PEAK_HOURS = delay_by_hour[delay_by_hour > delay_by_hour.mean()].index.tolist()
    
    df["IS_PEAK_HOUR"] = df["DEP_HOUR"].isin(PEAK_HOURS).astype(int)
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6,7]).astype(int)
    df["DISTANCE_BIN"] = pd.cut(
        df["DISTANCE"],
        bins=[0,500,1000,2000,5000],
        labels=["short","medium","long","very_long"]
    )

    print("Sampling yapılıyor (50k)...")
    df = df.sample(n=50_000, random_state=42)

    print("Train-test split ve preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

    # Pipeline: preprocessing + model
    print("Model oluşturuluyor...")
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    # Eğit
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)

     # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

    # Model klasörünü oluştur
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Modeli kaydet
    joblib.dump(model, MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")

    return model

#Kaydedilmiş modeli yükler.
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model bulunamadı, önce train_model() çalıştır.")
    return joblib.load(MODEL_PATH)

#Tek bir örnek için tahmin döner.
#input_dict: {"FLIGHT_NUMBER": 123, "ORIGIN": "JFK", ... }
def predict(input_dict):
    global PEAK_HOURS

    model = load_model()
    df = pd.DataFrame([input_dict])

    if PEAK_HOURS is None:
        raise ValueError("PEAK_HOURS not set. Run train_model() first.")

    # FEATURE ENGINEERING (AYNI TANIM!)
    df["IS_PEAK_HOUR"] = df["DEP_HOUR"].isin(PEAK_HOURS).astype(int)
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6,7]).astype(int)
    df["DISTANCE_BIN"] = pd.cut(
        df["DISTANCE"],
        bins=[0,500,1000,2000,5000],
        labels=["short","medium","long","very_long"]
    )

    return model.predict(df)[0]


if __name__ == "__main__":
    train_model()

