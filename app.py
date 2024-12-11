import streamlit as st
import joblib
import os
import requests
import streamlit as st
import sounddevice as sd
import whisper
from gtts import gTTS
from io import BytesIO
import tempfile
import numpy as np
from scipy.io.wavfile import write
import os

# Load Whisper Model (Tiny for faster processing)
model = whisper.load_model("tiny")

st.set_page_config(layout="wide")
# --- Function for Home Page ---
def home_page():
    st.title("Our Goal")
    st.write(
        """
        Adopting innovative techniques and leveraging advanced technologies, sustainable agriculture aims to enhance productivity, conserve natural resources, and promote long-term environmental stability.
        """
    )


# --- Function for Chatbot Page ---
def chatbot_page():
    # Predefined responses (Intent-Response Mapping)
    responses_ur = {
        "فصل کا مشورہ": "گندم اس وقت کاشت کے لئے بہترین ہے۔",
        "موسم": "آج موسم صاف رہے گا، کاشت کے لئے اچھا وقت ہے۔",
        "کیڈے مار دوا": "آپ نیم کا سپرے استعمال کر سکتے ہیں۔",
        "زمین کا معیار": "آپ کی زمین کا معیار اہم ہے، اس کے لئے آپ کو زمین کے پی ایچ اور نائٹروجن کی مقدار جانچنی چاہئے۔",
        "مٹی کی صحت": "مٹی کی صحت بہتر بنانے کے لئے، آپ کو آلیشیا اور ورمی کمپوسٹ کا استعمال کرنا چاہئے۔",
        "آبپاشی": "مٹی کی نمی کا خیال رکھنا ضروری ہے، آپ کو مناسب آبپاشی کی ضرورت ہوگی۔",
        "نائٹروجن کی کمی": "اگر مٹی میں نائٹروجن کی کمی ہے تو آپ کو یوریا کھاد کا استعمال کرنا چاہئے۔",
        "پانی کی کمی": "اگر پانی کی کمی ہو، تو آپ کو ڈرپ ایریگیشن سسٹم پر غور کرنا چاہئے تاکہ پانی کا ضیاع نہ ہو۔",
        "دھوپ اور درجہ حرارت": "پودوں کو دھوپ اور درجہ حرارت کے بارے میں آگاہی دینی ضروری ہے تاکہ بہتر پیداوار حاصل کی جا سکے۔"
    }

    responses_en = {
        "Hello": "Hi , I am Malhi Bot",
        "Crop Advice": "Wheat is the best crop for this season.",
        "Weather": "The weather will remain clear today, it's a good time for planting.",
        "Pesticide": "You can use neem spray.",
        "Soil Quality": "The quality of your soil is important, you should check the pH and nitrogen levels of the soil.",
        "Soil Health": "To improve soil health, you should use algae and vermicompost.",
        "Irrigation": "It's important to maintain soil moisture, you will need appropriate irrigation.",
        "Nitrogen Deficiency": "If there is a nitrogen deficiency in the soil, you should use urea fertilizer.",
        "Water Deficiency": "If there is a water shortage, consider using a drip irrigation system to avoid water wastage.",
        "Sunshine and Temperature": "It's important to educate plants about sunshine and temperature for better yield."
    }

    # Function for Speech-to-Text
    def speech_to_text(audio_path, language="ur"):
        result = model.transcribe(audio_path, language=language)
        return result["text"]

    # Function to detect intent (simple keyword matching)
    def detect_intent(transcribed_text, language="ur"):
        if language == "ur":
            if "فصل" in transcribed_text:
                return "فصل کا مشورہ"
            elif "موسم" in transcribed_text:
                return "موسم"
            elif "کیڈے" in transcribed_text:
                return "کیڈے مار دوا"
            elif "زمین" in transcribed_text:
                return "زمین کا معیار"
            elif "مٹی" in transcribed_text:
                return "مٹی کی صحت"
            elif "نائٹروجن" in transcribed_text:
                return "نائٹروجن کی کمی"
            elif "پانی" in transcribed_text:
                return "پانی کی کمی"
            elif "دھوپ" in transcribed_text:
                return "دھوپ اور درجہ حرارت"
            else:
                return None
        elif language == "en":
            if "crop" in transcribed_text:
                return "Crop Advice"
            elif "weather" in transcribed_text:
                return "Weather"
            elif "pesticide" in transcribed_text:
                return "Pesticide"
            elif "soil quality" in transcribed_text:
                return "Soil Quality"
            elif "soil health" in transcribed_text:
                return "Soil Health"
            elif "irrigation" in transcribed_text:
                return "Irrigation"
            elif "nitrogen deficiency" in transcribed_text:
                return "Nitrogen Deficiency"
            elif "water deficiency" in transcribed_text:
                return "Water Deficiency"
            elif "sunshine" in transcribed_text:
                return "Sunshine and Temperature"
            else:
                return None

    # Function for Text-to-Speech
    def text_to_speech(text, language="ur"):
        tts = gTTS(text, lang=language)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes

    # Function to record audio
    def record_audio(duration=5, samplerate=16000):
        st.write("Recording is starting...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        st.write("Recording has finished.")
        return audio, samplerate

    # Streamlit App
    st.title("Multilingual Speech-to-Speech Chatbot")
    st.write("Chatbot to assist farmers with information on crops, weather, soil health, etc.")

    # Language Switcher
    language = st.selectbox("Select Language", ["Urdu", "English"])
    language_code = "ur" if language == "Urdu" else "en"

    if st.button("Record Audio"):  # Record Button
        audio, samplerate = record_audio(duration=5)

        # Save audio to a temporary file in correct format
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            write(tmpfile.name, samplerate, audio)
            audio_path = tmpfile.name

        # Speech-to-Text
        try:
            transcribed_text = speech_to_text(audio_path, language=language_code)
            st.write("Transcribed Text:", transcribed_text)

            # Intent Detection
            intent = detect_intent(transcribed_text, language_code)
            if intent:
                st.write("Intent:", intent)
                response_text = responses_ur[intent] if language_code == "ur" else responses_en[intent]
            else:
                response_text = "Sorry, I couldn't understand that."

            # Text-to-Speech
            st.write("Response:", response_text)
            audio_response = text_to_speech(response_text, language_code)
            st.audio(audio_response, format="audio/mp3")
        except Exception as e:
            st.error(f"Error: {e}")

        # Clean up temporary audio file
        os.remove(audio_path)
# --- Function for Soil Prediction Page ---
def soil_prediction_page():
    st.title("Soil Monitoring and Prediction")
    st.write("This section helps predict soil conditions using AI models.")
    
    # User input for soil monitoring features
    air_temperature = st.number_input("Air Temperature (°C)", min_value=-50.0, max_value=50.0, value=28.9)
    soil_temperature = st.number_input("Soil Temperature (°C)", min_value=-50.0, max_value=50.0, value=26.81)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=42.0)
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=60.74)
    nitrogen = st.number_input("Nitrogen (ppm)", min_value=0.0, value=4.97)
    phosphorous = st.number_input("Phosphorous (ppm)", min_value=0.0, value=32.24)
    potassium = st.number_input("Potassium (ppm)", min_value=0.0, value=24.44)
    
    # Bundle input data into a list for prediction
    input_data = [air_temperature, soil_temperature, humidity, moisture, nitrogen, phosphorous, potassium]

    # Load the model
    model = load_model()

    # Make prediction when the user clicks the button
    if st.button("Predict Soil Health"):
        prediction = make_prediction(model, input_data)
        
        # Display prediction result
        if prediction == 0:
            st.write("Predicted Soil Health: **Poor Soil Health**")
        elif prediction == 1:
            st.write("Predicted Soil Health: **Moderate Soil Health**")
        else:
            st.write("Predicted Soil Health: **Good Soil Health**")

# --- Load the pre-trained model ---
def load_model():
    # Use a relative path to ensure portability in deployment
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(model_path)
    return model

# --- Function to make prediction ---
def make_prediction(model, input_data):
    prediction = model.predict([input_data])
    return prediction

def navbar():
    st.title("EarthRenewal.AI")  # Display the project name
    st.sidebar.title('Navigation')  # Sidebar title

    # Sidebar navigation buttons
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Chatbot", "Soil Monitoring", "Contact Us"]
    )
    
    return page

# --- Function to Handle Theme Toggle ---
def theme_toggle():
    theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
        <style>
            body {
                background-color: #333;
                color: black;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            body {
                background-color: #fafafa;
                color: black;
            }
        </style>
        """, unsafe_allow_html=True)

# # --- Function to Create a Custom Header and Navigation Bar ---
# hide_st_style = """
#             <style>
#             #MainMenu {visibilty: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
# def navbar():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #fff;
#         }
#         .header {
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#             background-color: #2d6a4f;
#             padding: 10px;
#             color: #edecec;
#             position: fixed;
#             top: 0;
#             left:0;
#             width: 100%;
#             z-index: 1000;
#             padding-right: 50px;
#             margin=0;
#             box-sizing: border-box;
#         }
#         .header img {
#             height: 50px;
#         }
#         .header .project-name {
#             font-size: 24px;
#             font-weight: bold;
#             margin-left: 10px;
#         }
#         .header .nav-links {
#             display: flex;
#             gap: 15px;
#         }
#         .header .nav-links a {
#             color: white;
#             text-decoration: none;
#             font-size: 16px;
#         }
#         .header .nav-links a:hover {
#             text-decoration: underline;
#         }
#     .navbar-buttons {
#             display: flex;
#             gap: 10px;
#             margin-top: 10px;
#         }
#         .navbar-buttons button {
#             padding: 8px 16px;
#             font-size: 16px;
#             background-color: #2d6a4f;
#             color: white;
#             border: none;
#             cursor: pointer;
#             transition: all 0.3s ease;
#         }
#         .navbar-buttons button:hover {
#             background-color: #d6e865;
#         }
#         .navbar-buttons button:active {
#             background-color: #9edd6c;
#         }
#         .navbar-buttons button.selected {
#             background-color: #9edd6c;
#         }
#     </style>
#     <div class="header">
#         <div style="display: flex; align-items: center;">
#             <img src="Frontend/logo.png" alt="Logo" style="height:50px;">
#             <span class="project-name">EarthRenewal.AI</span>
#         </div>
#         <div class="navbar-buttons">
#             <button id="homeBtn">Home</button>
#             <button id="chatbotBtn">Chatbot</button>
#             <button id="soilBtn">Soil Monitoring</button>
#             <button id="contactBtn">Contact Us</button>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # Create buttons for navbar using Streamlit's button widget
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         if st.button('Home'):
#             st.session_state.page = "Home"
#             st.markdown('<style> #homeBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)
#     with col2:
#         if st.button('Chatbot'):
#             st.session_state.page = "Chatbot"
#             st.markdown('<style> #chatbotBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)
#     with col3:
#         if st.button('Soil Monitoring'):
#             st.session_state.page = "Soil Prediction"
#             st.markdown('<style> #soilBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)
#     with col4:
#         if st.button('Contact Us'):
#             st.session_state.page = "Contact Us"
#             st.markdown('<style> #contactBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)

# --- Main Function to Handle Navigation ---
def main():
    theme_toggle()  # Display the theme toggle
    page = navbar()  # Get the selected page from navbar

    if page == "Home":
        home_page()
    elif page == "Chatbot":
        chatbot_page()
    elif page == "Soil Monitoring":
        soil_prediction_page()
    elif page == "Contact Us":
        contact_page()

# --- Run the Streamlit app ---
if __name__ == "__main__":
    main()