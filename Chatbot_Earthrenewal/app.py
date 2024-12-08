# # Main.py (Pipeline)

# from TranC import record_audio, transcribe_audio, translate_urdu_to_english, translate_english_to_urdu, text_to_speech
# from ChatC import user_input



# def pipeline():
#     # Step 1: Record audio from the user
#     audio_file = "recorded_audio.wav"
#     duration = 10  # 10 seconds of audio (can be modified as needed)
#     record_audio(audio_file, duration)

#     # Step 2: Transcribe the audio to Urdu text
#     transcribed_text = transcribe_audio(audio_file)
#     print(f"Transcribed text from audio: {transcribed_text}")

#     # Step 3: Translate the transcribed Urdu text to English
#     translated_english = translate_urdu_to_english(transcribed_text)
#     print(f"Translated to English: {translated_english}")

#     # Step 4: Get user input (you can modify this part for different purposes)
    
#     output_llm = user_input(translated_english)
    
#     # user_english_input = user_input()
    
#     print(f"LLM: {output_llm}")

#     # Step 5: Translate the user's English input back to Urdu
#     final_translated_urdu = translate_english_to_urdu(output_llm)
#     text_to_speech(final_translated_urdu)
    
#     print(f"Final output translated to Urdu: {final_translated_urdu}")

# # Run the pipeline
# if __name__ == "__main__":
#     pipeline()



from flask import Flask, request, jsonify
from TranC import record_audio, transcribe_audio, translate_urdu_to_english, translate_english_to_urdu, text_to_speech
from ChatC import user_input

app = Flask(__name__)

@app.route('/pipeline', methods=['POST'])
def run_pipeline():
    """
    Run the full pipeline: Record -> Transcribe -> Translate -> User Input -> Translate Back -> TTS
    """
    try:
        # Step 1: Get duration from the request (default to 10 seconds)
        data = request.json
        duration = data.get('duration', 10)  # Default to 10 seconds

        # Step 2: Record audio
        audio_file = "recorded_audio.wav"
        record_audio(audio_file, duration)

        # Step 3: Transcribe the audio
        transcribed_text = transcribe_audio(audio_file)
        print(f"Transcribed text: {transcribed_text}")

        # Step 4: Translate Urdu to English
        translated_english = translate_urdu_to_english(transcribed_text)
        print(f"Translated to English: {translated_english}")

        # Step 5: Process user input via ChatC
        output_llm = user_input(translated_english)
        print(f"LLM Output: {output_llm}")

        # Step 6: Translate back to Urdu
        final_translated_urdu = translate_english_to_urdu(output_llm)

        # Step 7: Convert Urdu text to speech
        text_to_speech(final_translated_urdu)
        print(f"Final output in Urdu: {final_translated_urdu}")

        # Step 8: Return response
        return jsonify({
            "transcribed_text": transcribed_text,
            "translated_english": translated_english,
            "llm_output": output_llm,
            "final_translated_urdu": final_translated_urdu
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
