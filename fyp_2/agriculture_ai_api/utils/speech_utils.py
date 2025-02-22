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
