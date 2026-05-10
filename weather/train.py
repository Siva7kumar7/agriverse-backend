"""
ADVANCED WEATHER RAIN PREDICTION – HIGH ACCURACY
Target: rain_tomorrow
Model: Optimized Random Forest (Rain-focused)
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# =============================
# PATH CONFIG
# =============================
WEATHER_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(WEATHER_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "tamilnadu_weather.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset Loaded:", df.shape)

# =============================
# CLEANING
# =============================
df.drop_duplicates(inplace=True)
df.ffill(inplace=True)

# =============================
# TIME FEATURE ENGINEERING
# =============================
df["time"] = pd.to_datetime(df["time"], errors="coerce")

df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

df.drop(columns=["time"], inplace=True)

# =============================
# RAIN INTENSITY FEATURE (KEY)
# =============================
df["rain_intensity"] = (
    df["precipitation"] * df["cloud_cover"] * (df["relative_humidity_2m"] / 100)
)

# =============================
# CITY ENCODING
# =============================
city_encoder = LabelEncoder()
df["city"] = city_encoder.fit_transform(df["city"])

joblib.dump(city_encoder, os.path.join(MODEL_DIR, "city_encoder.pkl"))

# =============================
# FEATURES & TARGET
# =============================
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_direction_10m",
    "rain_intensity",
    "hour",
    "day",
    "month",
    "dayofweek",
    "is_weekend",
    "city"
]

TARGET = "rain_tomorrow"

X = df[FEATURES]
y = df[TARGET]

print("\n☔ Rain Tomorrow Distribution")
print(y.value_counts(normalize=True))

# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =============================
# SCALING
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# =============================
# RANDOM FOREST (RAIN-FOCUSED)
# =============================
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight={0: 1, 1: 3},   # 🔥 prioritize rain detection
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# =============================
# EVALUATION
# =============================
probs = rf_model.predict_proba(X_test)[:, 1]

# Lower threshold → detect rain better
THRESHOLD = 0.45
preds = (probs >= THRESHOLD).astype(int)

print("\n🌧️ RANDOM FOREST PERFORMANCE (RAIN-OPTIMIZED)")
print("Accuracy:", accuracy_score(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds))

joblib.dump(rf_model, os.path.join(MODEL_DIR, "rain_rf.pkl"))

print("\n✅ HIGH ACCURACY WEATHER MODEL TRAINED")
print("📁 Saved in backend/models/")
