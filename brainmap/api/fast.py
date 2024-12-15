import sys
import os

# Add the parent directory of 'brainmap' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, File, UploadFile, HTTPException
from brainmap.ml_logic.registry import load_model
from brainmap.ml_logic.preprocessor import preprocess_image
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
app.state.model = load_model("brainmap/api/checkpoints/best_model.keras")

class_names = np.array(['glioma', 'meningioma', 'pituitary', 'notumor'])

@app.get("/")
def root():
    return {'Root': "Welcome to BrainMap API"}

@app.post("/predict")
async def create_upload_file(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="No upload file sent")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Preprocess the image
        preprocessed_image = preprocess_image(image, target_size=(128, 128))

        prediction = app.state.model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        probability = round(float(np.max(prediction)), 4)

        return {"Prediction": predicted_class, "Probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
