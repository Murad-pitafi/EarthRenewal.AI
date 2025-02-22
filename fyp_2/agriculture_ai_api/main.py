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
