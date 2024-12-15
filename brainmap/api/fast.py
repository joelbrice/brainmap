from fastapi import FastAPI, File, UploadFile
import pandas as pd
from brainmap.ml_logic.registry import load_model
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
app.state.model = load_model()

@app.get("/")
def root():
    return {'Root': "Welcome to BrainMap API"}

@app.post("/predict")
async def create_upload_file(file: UploadFile):
    if not file:
        return {"message": "No upload file sent"}
    else:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        expanded_array_img = np.expand_dims(image, axis=0)
        prediction = app.state.model.predict(expanded_array_img)
        probability = app.state.model.predict_proba(expanded_array_img)
        pred = round(float(prediction[0][0]), 4)
        probability = round(float(probability[0][0]), 4)

        return {"Prediction": pred, "Probability": probability}
