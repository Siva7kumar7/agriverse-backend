"""
Weather API Integration Module
Fetches real-time weather data from OpenWeatherMap API
"""
import os
import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# OpenWeatherMap API configuration
API_KEY = "1b09d0bfc92c612b9635e2482831ddc9"
BASE_URL = "http://api.openweathermap.org/data/2.5"
GEOCODING_URL = "http://api.openweathermap.org/geo/1.0"

def get_coordinates(city_name: str, state_code: str = "", country_code: str = "IN") -> Optional[Dict]:
    """
    Get latitude and longitude for a city using OpenWeatherMap Geocoding API
    
    Args:
        city_name: Name of the city
        state_code: State code (optional)
        country_code: Country code (default: IN for India)
        
    Returns:
        Dictionary with 'lat' and 'lon' or None if failed
    """
    if not API_KEY:
        logger.warning("OpenWeatherMap API key not set. Set OPENWEATHER_API_KEY environment variable.")
        return None
    
    try:
        url = f"{GEOCODING_URL}/direct"
        params = {
            'q': f"{city_name},{state_code},{country_code}".strip(','),
            'limit': 1,
            'appid': API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            return {
                'lat': data[0]['lat'],
                'lon': data[0]['lon'],
                'name': data[0]['name'],
                'state': data[0].get('state', ''),
                'country': data[0].get('country', '')
            }
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching coordinates: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_coordinates: {e}")
        return None

def get_current_weather(city_name: str = None, lat: float = None, lon: float = None) -> Optional[Dict]:
    """
    Fetch current weather data from OpenWeatherMap API
    
    Args:
        city_name: Name of the city (if lat/lon not provided)
        lat: Latitude (optional)
        lon: Longitude (optional)
        
    Returns:
        Dictionary with weather data in format compatible with prediction model
        Returns None if API key not set or request fails
    """
    if not API_KEY:
        logger.warning("OpenWeatherMap API key not set. Using default values.")
        return None
    
    try:
        # Get coordinates if city name provided
        if city_name and (lat is None or lon is None):
            coords = get_coordinates(city_name)
            if coords:
                lat = coords['lat']
                lon = coords['lon']
                logger.info(f"Found coordinates for {city_name}: lat={lat}, lon={lon}")
            else:
                logger.warning(f"Could not find coordinates for {city_name}")
                return None
        
        if lat is None or lon is None:
            logger.error("Either city_name or lat/lon must be provided")
            return None
        
        # Fetch current weather
        url = f"{BASE_URL}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': 'metric'  # Get temperature in Celsius
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract and format data for our prediction model
        weather_data = {
            'city': data.get('name', city_name or 'Unknown'),
            'temperature_2m': data['main']['temp'],
            'relative_humidity_2m': data['main']['humidity'],
            'dew_point_2m': data['main']['temp'] - ((100 - data['main']['humidity']) / 5),  # Approximate dew point
            'surface_pressure': data['main']['pressure'],  # hPa
            'cloud_cover': data['clouds']['all'],  # Percentage
            'cloud_cover_low': data['clouds']['all'],  # Using same value (API doesn't provide separate)
            'wind_speed_10m': data['wind']['speed'] * 3.6,  # Convert m/s to km/h
            'wind_direction_10m': data['wind'].get('deg', 0),
            'condition': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
            'timestamp': data['dt']
        }
        
        logger.info(f"Successfully fetched weather data for {weather_data['city']}")
        return weather_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return None
    except KeyError as e:
        logger.error(f"Missing key in weather data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_current_weather: {e}")
        return None

def get_weather_forecast(city_name: str = None, lat: float = None, lon: float = None, days: int = 7) -> Optional[Dict]:
    """
    Fetch weather forecast from OpenWeatherMap API
    
    Args:
        city_name: Name of the city
        lat: Latitude
        lon: Longitude
        days: Number of days to forecast (max 5 for free tier)
        
    Returns:
        Dictionary with forecast data
    """
    if not API_KEY:
        return None
    
    try:
        # Get coordinates if needed
        if city_name and (lat is None or lon is None):
            coords = get_coordinates(city_name)
            if coords:
                lat = coords['lat']
                lon = coords['lon']
            else:
                return None
        
        if lat is None or lon is None:
            return None
        
        # Fetch forecast (5-day forecast available in free tier)
        url = f"{BASE_URL}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': API_KEY,
            'units': 'metric',
            'cnt': min(days * 8, 40)  # 8 forecasts per day, max 40 for free tier
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        return None

