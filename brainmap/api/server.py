from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

# Verify the model file exists
model_path = "cnn_model.keras"
print(f"Looking for model at: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No such file or directory: '{model_path}'")

# Load the model using TensorFlow/Keras
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

class_names = np.array(['glioma', 'meningioma', 'pituitary', 'notumor'])

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

@app.get("/")
def brainmap():
    return {"message": "Welcome to the BrainMap API!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape((1, 256, 256, 3))  # Adjust shape as needed
        prediction = model.predict(features)
        predicted_class = class_names[np.argmax(prediction)]
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
