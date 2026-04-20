import json
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_image
from disease_info import DISEASE_INFO

MODEL_PATH = "models/skin_model.h5"
CLASS_FILE = "class_names.json"

model = load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

def predict_skin_disease(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    predicted_label = class_names[str(predicted_index)]

    disease_details = DISEASE_INFO.get(predicted_label, {
        "description": "No description available.",
        "precautions": []
    })

    warning = ""
    if confidence < 60:
        warning = "Low confidence prediction. Manual review by doctor is recommended."

    return {
        "predicted_class": predicted_label,
        "confidence": round(confidence, 2),
        "description": disease_details["description"],
        "precautions": disease_details["precautions"],
        "warning": warning
    }