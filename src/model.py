import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import joblib

# Path Ayarları
DATA_PATH = os.path.join("..", "data", "flights.csv")
MODEL_PATH = os.path.join("..", "models", "flight_delay_model.pkl")

# Veriyi yükler ve DataFrame döner.
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No data found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

# Feature/target ayırır ve train/test split yapar.
# Ayrıca preprocessing pipeline’ı da döndürür.
def preprocess_and_split(df):
    target = "ARRIVAL_DELAY"   # dataset'teki hedef değişken
    X = df.drop(columns=[target])
    y = df[target]

# Kategorik & sayısal kolonlar
categorical_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
numeric_cols = ["MONTH", "DAY", "DAY_OF_WEEK", "DISTANCE", "DEP_HOUR"]

# Preprocessing pipeline (One-hot encoding)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Train-test split
 X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, preprocessor

# Modeli eğitir ve models/ altına kaydeder.
def train_model():
    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

    # Pipeline: preprocessing + model
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    # Eğit
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)

    # Modeli kaydet
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model kaydedildi: {MODEL_PATH}")

    # performans raporu
    score = model.score(X_test, y_test)
    print(f"Test R^2 Score: {score:.4f}")

    return model

#Kaydedilmiş modeli yükler.
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model bulunamadı, önce train_model() çalıştır.")
    return joblib.load(MODEL_PATH)

#Tek bir örnek için tahmin döner.
#input_dict: {"FLIGHT_NUMBER": 123, "ORIGIN": "JFK", ... }
def predict(input_dict):
    model = load_model()
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    return prediction

# Accuracy ölç
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)