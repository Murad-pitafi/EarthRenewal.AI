# # import streamlit as st
# # from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
# # import whisper
# # from gtts import gTTS
# # from io import BytesIO
# # import os
# # import tempfile

# # # Load Whisper Model (Tiny for faster processing)
# # model = whisper.load_model("tiny")

# # # Predefined responses (Intent-Response Mapping)
# # responses = {
# #     "فصل کا مشورہ": "گندم اس وقت کاشت کے لیے بہترین ہے۔",
# #     "موسم": "آج موسم صاف رہے گا، کاشت کے لیے اچھا وقت ہے۔",
# #     "کیڑے مار دوا": "آپ نیم کا سپرے استعمال کر سکتے ہیں۔"
# # }

# # # Function for Speech-to-Text
# # def speech_to_text(audio_path):
# #     result = model.transcribe(audio_path, language="ur")
# #     return result["text"]

# # # Function to detect intent (simple keyword matching)
# # def detect_intent(transcribed_text):
# #     if "فصل" in transcribed_text:
# #         return "فصل کا مشورہ"
# #     elif "موسم" in transcribed_text:
# #         return "موسم"
# #     elif "کیڑے" in transcribed_text:
# #         return "کیڑے مار دوا"
# #     else:
# #         return None

# # # Function for Text-to-Speech
# # def text_to_speech(text):
# #     tts = gTTS(text, lang="ur")
# #     audio_bytes = BytesIO()
# #     tts.write_to_fp(audio_bytes)
# #     audio_bytes.seek(0)
# #     return audio_bytes

# # # Define an Audio Processor for recording
# # class AudioProcessor(AudioProcessorBase):
# #     def __init__(self):
# #         self.recorded_audio = None

# #     def recv_audio(self, frame):
# #         self.recorded_audio = frame.to_ndarray()
# #         return frame

# # # Streamlit App
# # st.title("Multilingual Speech-to-Speech Chatbot")
# # st.write("اردو زبان میں کسانوں کے لئے معاون چیٹ بوٹ")

# # # Real-time Audio Recording
# # st.write("اپنی آواز ریکارڈ کریں:")
# # webrtc_ctx = webrtc_streamer(
# #     key="speech_recorder",
# #     mode=WebRtcMode.SENDRECV,
# #     audio_receiver_size=256,
# #     media_stream_constraints={"audio": True},
# #     audio_processor_factory=AudioProcessor,
# # )

# # if webrtc_ctx and webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.recorded_audio:
# #     st.write("ریکارڈ شدہ آڈیو پر کارروائی ہو رہی ہے...")
    
# #     # Save audio to a temporary file
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
# #         tmpfile.write(webrtc_ctx.audio_processor.recorded_audio.tobytes())
# #         audio_path = tmpfile.name

# #     # Speech-to-Text
# #     transcribed_text = speech_to_text(audio_path)
# #     st.write("متن:", transcribed_text)

# #     # Intent Detection
# #     intent = detect_intent(transcribed_text)
# #     if intent:
# #         st.write("ارادہ:", intent)
# #         response_text = responses[intent]
# #     else:
# #         response_text = "معذرت، میں آپ کی بات کو سمجھ نہیں سکا۔"
    
# #     # Text-to-Speech
# #     st.write("جواب:", response_text)
# #     audio_response = text_to_speech(response_text)
# #     st.audio(audio_response, format="audio/mp3")

# #     # Clean up temporary audio file
# #     os.remove(audio_path)
# import streamlit as st
# import sounddevice as sd
# import whisper
# from gtts import gTTS
# from io import BytesIO
# import tempfile
# import numpy as np
# import os

# # Load Whisper Model (Tiny for faster processing)
# model = whisper.load_model("tiny")

# # Predefined responses (Intent-Response Mapping)
# responses = {
#     "فصل کا مشورہ": "گندم اس وقت کاشت کے لئے بہترین ہے۔",
#     "موسم": "آج موسم صاف رہے گا، کاشت کے لئے اچھا وقت ہے۔",
#     "کیڈے مار دوا": "آپ نیم کا سپرے استعمال کر سکتے ہیں۔"
# }

# # Function for Speech-to-Text
# def speech_to_text(audio_path):
#     result = model.transcribe(audio_path, language="ur")
#     return result["text"]

# # Function to detect intent (simple keyword matching)
# def detect_intent(transcribed_text):
#     if "فصل" in transcribed_text:
#         return "فصل کا مشورہ"
#     elif "موسم" in transcribed_text:
#         return "موسم"
#     elif "کیڈے" in transcribed_text:
#         return "کیڈے مار دوا"
#     else:
#         return None

# # Function for Text-to-Speech
# def text_to_speech(text):
#     tts = gTTS(text, lang="ur")
#     audio_bytes = BytesIO()
#     tts.write_to_fp(audio_bytes)
#     audio_bytes.seek(0)
#     return audio_bytes

# # Function to record audio
# def record_audio(duration=5, samplerate=16000):
#     st.write("ریکارڈنگ شروع ہو رہی ہے...")
#     audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
#     sd.wait()  # Wait until recording is finished
#     st.write("ریکارڈنگ ختم ہو گئی۔")
#     return audio, samplerate

# # Streamlit App
# st.title("Multilingual Speech-to-Speech Chatbot")
# st.write("اردو زبان میں کسانوں کے لئے معاون چیٹ بوٹ")

# if st.button("آواز ریکارڈ کریں"):  # Record Button
#     audio, samplerate = record_audio(duration=5)

#     # Save audio to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#         tmpfile.write(audio.tobytes())
#         audio_path = tmpfile.name

#     # Speech-to-Text
#     transcribed_text = speech_to_text(audio_path)
#     st.write("متن:", transcribed_text)

#     # Intent Detection
#     intent = detect_intent(transcribed_text)
#     if intent:
#         st.write("ارادہ:", intent)
#         response_text = responses[intent]
#     else:
#         response_text = "معذرت، میں آپ کی بات کو سمجھ نہیں سکا۔"

#     # Text-to-Speech
#     st.write("جواب:", response_text)
#     audio_response = text_to_speech(response_text)
#     st.audio(audio_response, format="audio/mp3")

#     # Clean up temporary audio file
#     os.remove(audio_path)
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

# Predefined responses (Intent-Response Mapping)
# Predefined responses (Intent-Response Mapping)
responses = {
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


# Function for Speech-to-Text
def speech_to_text(audio_path):
    result = model.transcribe(audio_path, language="ur")
    return result["text"]

# Function to detect intent (simple keyword matching)
def detect_intent(transcribed_text):
    if "فصل" in transcribed_text:
        return "فصل کا مشورہ"
    elif "موسم" in transcribed_text:
        return "موسم"
    elif "کیڈے" in transcribed_text:
        return "کیڈے مار دوا"
    elif "زمین":
        return "زمین کا معیار"    
    elif "مٹی":
        return "مٹی کی صحت"
    elif "نائٹروجن":
        return "نائٹروجن کی کمی"
    elif "پانی":
        return "پانی کی کمی"
    elif "دھوپ":
        return "دھوپ اور درجہ حرارت"        
    else:
        return None

# Function for Text-to-Speech
def text_to_speech(text):
    tts = gTTS(text, lang="ur")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    st.write("ریکارڈنگ شروع ہو رہی ہے...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.write("ریکارڈنگ ختم ہو گئی۔")
    return audio, samplerate

# Streamlit App
st.title("Multilingual Speech-to-Speech Chatbot")
st.write("اردو زبان میں کسانوں کے لئے معاون چیٹ بوٹ")

if st.button("آواز ریکارڈ کریں"):  # Record Button
    audio, samplerate = record_audio(duration=5)

    # Save audio to a temporary file in correct format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write(tmpfile.name, samplerate, audio)
        audio_path = tmpfile.name

    # Speech-to-Text
    try:
        transcribed_text = speech_to_text(audio_path)
        st.write("متن:", transcribed_text)

        # Intent Detection
        intent = detect_intent(transcribed_text)
        if intent:
            st.write("ارادہ:", intent)
            response_text = responses[intent]
        else:
            response_text = "معذرت، میں آپ کی بات کو سمجھ نہیں سکا۔"

        # Text-to-Speech
        st.write("جواب:", response_text)
        audio_response = text_to_speech(response_text)
        st.audio(audio_response, format="audio/mp3")
    except Exception as e:
        st.error(f"خرابی: {e}")

    # Clean up temporary audio file
    os.remove(audio_path)
