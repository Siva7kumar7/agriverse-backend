# weather/predict.py

import os
import numpy as np
import joblib
import logging
from weather.weather_api import get_current_weather

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rain_rf.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_weather_models():
    global rf_model, scaler
    if rf_model is not None:
        return True
    try:
        if os.path.exists(RF_MODEL_PATH) and os.path.exists(SCALER_PATH):
            rf_model = joblib.load(RF_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("✅ Weather ML models loaded")
            return True
        else:
            print("⚠️ Weather model files missing, skipping ML prediction")
            return False
    except Exception as e:
        print("❌ Weather model load error:", e)
        return False


def predict_rainfall(lat, lon):
    try:
        load_weather_models()
        weather_data = get_current_weather(lat=lat, lon=lon)

        if not weather_data:
            return {
                "success": False,
                "error": "Weather API failed"
            }

        # Feature vector (IMPORTANT)
        features = np.array([[
            weather_data["temperature_2m"],
            weather_data["relative_humidity_2m"],
            weather_data["surface_pressure"],
            weather_data["wind_speed_10m"],
            weather_data["cloud_cover"]
        ]])

        if scaler:
            features = scaler.transform(features)

        if rf_model:
            prob = rf_model.predict_proba(features)[0][1] * 100
        else:
            prob = 0

        prob = round(prob, 1)

        if prob > 70:
            alert = "🌧 Heavy Rain Expected"
        elif prob > 40:
            alert = "🌦 Moderate Rain Possible"
        else:
            alert = "🌤 No Rain Expected"

        return {
            "success": True,
            "city": weather_data["city"],
            "temperature": weather_data["temperature_2m"],
            "humidity": weather_data["relative_humidity_2m"],
            "wind_speed": weather_data["wind_speed_10m"],
            "wind_direction": weather_data["wind_direction_10m"],
            "rain_probability": prob,
            "alert": alert
        }

    except Exception as e:
        logger.exception("Prediction error")
        return {
            "success": False,
            "error": str(e)
        }