import joblib
import numpy as np

model = joblib.load("models/crop_model.pkl")  # Ensure you have a trained model saved

def predict_crop(features):
    features_array = np.array([list(features.values())])
    return model.predict(features_array)[0]
