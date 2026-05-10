from flask import Blueprint, request, jsonify
from weather.predict import predict_rainfall
from weather.weather_api import get_weather_forecast
from datetime import datetime
import logging

weather_bp = Blueprint("weather", __name__)
logger = logging.getLogger(__name__)

# ================= WEATHER =================
@weather_bp.route("/weather/predict", methods=["POST"])
def weather_predict():
    data = request.get_json()
    lat = data.get("lat")
    lon = data.get("lon")

    result = predict_rainfall(float(lat), float(lon))
    return jsonify(result)


# ================= FORECAST =================
@weather_bp.route("/weather/forecast", methods=["POST"])
def forecast_weather():
    try:
        data = request.get_json()
        lat = data.get("lat")
        lon = data.get("lon")

        res = get_weather_forecast(lat=lat, lon=lon)

        if not res:
            return jsonify({"forecast": []})

        forecast_list = res["list"]

        daily = {}

        for item in forecast_list:
            date = item["dt_txt"].split(" ")[0]

            if date not in daily:
                daily[date] = {
                    "temp": [],
                    "humidity": [],
                    "rain": [],
                    "icon": item["weather"][0]["icon"]
                }

            daily[date]["temp"].append(item["main"]["temp"])
            daily[date]["humidity"].append(item["main"]["humidity"])
            daily[date]["rain"].append(item.get("rain", {}).get("3h", 0))

        forecast = []
        for date, values in list(daily.items())[:7]:
            forecast.append({
                "day": datetime.strptime(date, "%Y-%m-%d").strftime("%A"),
                "temp": round(sum(values["temp"]) / len(values["temp"]), 1),
                "humidity": int(sum(values["humidity"]) / len(values["humidity"])),
                "rain": round(sum(values["rain"]), 1),
                "icon": f"https://openweathermap.org/img/wn/{values['icon']}@2x.png"
            })

        return jsonify({"forecast": forecast})

    except Exception as e:
        logger.exception("Forecast error")
        return jsonify({"forecast": []})