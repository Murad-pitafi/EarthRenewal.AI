#!/bin/bash

# Set project name
PROJECT_NAME="agriculture_ai_api"

# Create project directory
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

echo "📁 Creating project structure..."

# Create subdirectories
mkdir -p utils models temp

# Create Python files
touch main.py
touch utils/__init__.py
touch utils/speech_utils.py
touch utils/intent_utils.py
touch utils/model_utils.py

# Create README
cat <<EOL > README.md
# 🌾 Agriculture AI API

This project provides an AI-powered API for speech-to-text, text-to-speech, intent detection, and crop prediction.

## 🚀 Running the API
\`\`\`bash
python3 main.py
\`\`\`

## Endpoints
- **/speech-to-text/** - Convert speech to text
- **/detect-intent/** - Detect intent from text
- **/predict-crop/** - Predict suitable crops
- **/text-to-speech/** - Convert text to speech
EOL

# Create requirements.txt
echo "fastapi
uvicorn
whisper
gtts
joblib
numpy
pandas
scikit-learn" > requirements.txt

# Writing Python Code for APIs

# ✅ Main API file
cat <<EOL > main.py
from fastapi import FastAPI, UploadFile, File
import whisper
from utils.speech_utils import speech_to_text, text_to_speech
from utils.intent_utils import detect_intent
from utils.model_utils import predict_crop

app = FastAPI()

@app.post("/speech-to-text/")
async def convert_speech_to_text(file: UploadFile = File(...)):
    return {"text": speech_to_text(await file.read())}

@app.post("/detect-intent/")
async def get_intent(text: str):
    return {"intent": detect_intent(text)}

@app.post("/predict-crop/")
async def get_crop_prediction(features: dict):
    return {"prediction": predict_crop(features)}

@app.post("/text-to-speech/")
async def convert_text_to_speech(text: str):
    file_path = text_to_speech(text)
    return {"audio_file": file_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOL

# ✅ Speech-to-Text and Text-to-Speech
cat <<EOL > utils/speech_utils.py
import whisper
from gtts import gTTS
import os

model = whisper.load_model("tiny")

def speech_to_text(audio_bytes):
    with open("temp/audio.wav", "wb") as f:
        f.write(audio_bytes)
    result = model.transcribe("temp/audio.wav")
    return result["text"]

def text_to_speech(text):
    tts = gTTS(text)
    file_path = "temp/speech.mp3"
    tts.save(file_path)
    return file_path
EOL

# ✅ Intent Detection (Dummy Example)
cat <<EOL > utils/intent_utils.py
def detect_intent(text):
    keywords = {"weather": "Weather Inquiry", "crop": "Crop Inquiry"}
    for word, intent in keywords.items():
        if word in text.lower():
            return intent
    return "General Inquiry"
EOL

# ✅ Crop Prediction Model (Dummy Example)
cat <<EOL > utils/model_utils.py
import joblib
import numpy as np

model = joblib.load("models/crop_model.pkl")  # Ensure you have a trained model saved

def predict_crop(features):
    features_array = np.array([list(features.values())])
    return model.predict(features_array)[0]
EOL

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 FastAPI server is ready! Run it with: python3 main.py"
