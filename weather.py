# import requests

# # Your API Key
# API_KEY = "f9c06590b03947f8b0afaf4d4af135c3"  # Replace with your actual API key

# # Function to get weather data
# def get_weather(lat, lon, city_name):
#     url = f"https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={API_KEY}&include=daily"

#     response = requests.get(url)  # Send request to API
    
    

#     if response.status_code == 200:
#         data = response.json()  # Convert response to JSON
#         weather_data = data["data"][0]  # Extract the first result

#         # Extract required weather details
#         temperature = weather_data["temp"]  # Temperature in °C
#         humidity = weather_data["rh"]  # Relative humidity
#         condition = weather_data["weather"]["description"]  # Weather description

#         print(f"🌍 Weather in {city_name}:")
#         print(f"🌡 Temperature: {temperature}°C")
#         print(f"💧 Humidity: {humidity}%")
#         print(f"⛅ Condition: {condition}")
#     else:
#         print("❌ Error fetching weather data. Check your API key and location.")

# # Select a city in Pakistan
# city_name = "Karachi"
# latitude = 31.5497
# longitude = 74.3436

# # Call the function with Lahore's coordinates
# get_weather(latitude, longitude, city_name)

import streamlit as st
import requests

# Your API Key (Replace with your actual API key)
API_KEY = "f9c06590b03947f8b0afaf4d4af135c3"

# Dictionary of cities in Pakistan with lat/lon
PAKISTAN_CITIES = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Islamabad": (33.6844, 73.0479),
    "Faisalabad": (31.4504, 73.1350),
    "Multan": (30.1575, 71.5249),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750)
}

# Function to fetch weather data
def fetch_weather_data(city):
    lat, lon = PAKISTAN_CITIES[city]  # Get city coordinates
    url = f"https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={API_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_data = data["data"][0]

        return {
            "city": city,
            "temperature": weather_data["temp"],
            "humidity": weather_data["rh"],
            "condition": weather_data["weather"]["description"]
        }
    else:
        return {"error": "❌ Unable to fetch weather data. Check API key or location."}

# Function to display weather info (called in app.py)
def get_weather():
    st.title("🌾 Agriculture Weather Forecast - Pakistan 🇵🇰")

    # Select city from dropdown
    selected_city = st.selectbox("📍 Select a city:", list(PAKISTAN_CITIES.keys()))

    # Fetch and display weather data
    if st.button("🔍 Get Weather Info"):
        weather_info = fetch_weather_data(selected_city)

        if "error" in weather_info:
            st.error(weather_info["error"])
        else:
            st.success(f"🌍 **Weather in {weather_info['city']}**")
            st.write(f"🌡 **Temperature:** {weather_info['temperature']}°C")
            st.write(f"💧 **Humidity:** {weather_info['humidity']}%")
            st.write(f"⛅ **Condition:** {weather_info['condition']}")
