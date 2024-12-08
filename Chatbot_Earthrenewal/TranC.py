import whisper
from googletrans import Translator
import wave
import pyaudio
from gtts import gTTS
from playsound import playsound
import pygame


translator = Translator()


def record_audio(output_file, duration=10):
    # Set parameters for recording
    chunk = 1024  # Size of each audio chunk
    format = pyaudio.paInt16  # Audio format
    channels = 1  # Mono audio
    rate = 44100  # Sampling rate in Hz

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    # Record audio for the specified duration
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio to a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f"Recording saved as {output_file}")    
    
    
      
def transcribe_audio(file_path):
    print("Transcribing audio...")
    whisper_model = whisper.load_model("medium")
    result = whisper_model.transcribe(file_path, language='ur')
    return result['text']

def translate_urdu_to_english(urdu_text):
    
    # Step 2: Translate the transcribed Urdu text to English
    print("Translating Urdu to English...")
    translation = translator.translate(urdu_text, src='ur', dest='en')
    
    return translation.text


  
def translate_english_to_urdu(english_text):    
    print("Translating English to Urdu...")
    translation = translator.translate(english_text, src='en', dest='ur')

    return translation.text


def text_to_speech(urdu_text, audio_file="urdu_speech.mp3"):
    """
    Converts the given Urdu text to speech and plays the audio.
    
    Parameters:
    urdu_text (str): The Urdu text to convert to speech.
    audio_file (str): The name of the audio file to save and play. Default is 'urdu_speech.mp3'.
    """
    try:
        # Create TTS object and save audio
        tts = gTTS(text=urdu_text, lang='ur')
        tts.save(audio_file)
        print(f"Audio saved as {audio_file}")

        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        print("Playing audio...")
        while pygame.mixer.music.get_busy():  # Wait until playback is finished
            continue

        print("Audio playback finished.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main testing function
def test_all_functions():
    # Step 1: Record Audio
    output_file = "test_audio.wav"
    duration = 10  # 10 seconds of audio
    record_audio(output_file, duration)

    # Step 2: Transcribe Audio
    transcribed_text = transcribe_audio(output_file)
    print(f"Transcribed text: {transcribed_text}")

    # Step 3: Translate Transcribed Text (Urdu to English)
    translated_english = translate_urdu_to_english(transcribed_text)
    print(f"Translated to English: {translated_english}")

    # Step 4: Translate the English Text Back to Urdu
    translated_urdu = translate_english_to_urdu(translated_english)
    print(f"Translated back to Urdu: {translated_urdu}")

# Run the test function
