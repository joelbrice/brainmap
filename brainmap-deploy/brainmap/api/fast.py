import sys
import os
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from brainmap.ml_logic.registry import load_model
from brainmap.ml_logic.preprocessor import preprocess_image
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import uuid
import shap
import cv2
from google.cloud import storage


bucket_name="brainmap_bucket"

def upload_to_gcs(file_stream, bucket_name):
    """Upload file stream to Google Cloud Storage"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(f"images/{uuid.uuid4()}_yolo_detection.png")

    # Upload the in-memory file stream
    blob.upload_from_file(file_stream, content_type="image/png")

    # Make the file publicly accessible (optional)
    blob.make_public()

    return blob.public_url


# Initialize FastAPI
app = FastAPI()

# Load the CNN model and YOLO model at the start
@app.on_event("startup")
async def load_models():
    # Load the CNN model for prediction
    app.state.cnn_model = load_model("brainmap/api/checkpoints/best_model.keras")

    # Load the YOLO model for object detection
    yolo_checkpoint_dir = './brainmap/api/checkpoints'  # Correct path to checkpoints directory
    yolo_model_path = os.path.join(yolo_checkpoint_dir, 'yolo_model.pt')
    app.state.yolo_model = YOLO(yolo_model_path)


# Define class names for CNN
class_names = np.array(['glioma', 'meningioma', 'notumor', 'pituitary'])




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

        prediction = app.state.cnn_model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        probability = round(float(np.max(prediction)), 4)

        return {"Prediction": predicted_class, "Probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_yolo/")
async def predict_yolo(file: UploadFile = File(...)):
    """Predict using YOLO model for tumor detection and upload result to GCS"""
    try:
        # Save the uploaded file temporarily
        img_path = f"temp_{file.filename}"
        with open(img_path, "wb") as img_file:
            img_file.write(await file.read())

        # Perform YOLO detection
        results = app.state.yolo_model.predict(source=img_path)

        # Debug: Check the raw results and tensor shape
        print(f"YOLO results: {results}")

        # Check if results are available
        if len(results) == 0 or not results[0].boxes:
            raise ValueError("No valid detections found.")

        # Extract detections from the results
        detected_image = results[0].plot()

        # Upload the detected image to Google Cloud Storage
        yolo_image_path_in_memory = BytesIO()  # Use BytesIO to keep the image in memory
        plt.imshow(detected_image)
        plt.axis('off')
        plt.savefig(yolo_image_path_in_memory, format="png", bbox_inches='tight', pad_inches=0)
        yolo_image_path_in_memory.seek(0)  # Rewind the BytesIO object to the beginning

        # Upload the in-memory image to GCS
        yolo_image_url = upload_to_gcs(yolo_image_path_in_memory, bucket_name)

        # Cleanup temporary image file
        os.remove(img_path)

        return {
            "yolo_image_url": yolo_image_url,
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_cnn/")
async def explain_cnn_endpoint(file: UploadFile = File(...)):
    """Explain CNN predictions using SHAP"""
    try:
        # Save the uploaded file temporarily
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Preprocess the image
        preprocessed_image = preprocess_image(image, target_size=(128, 128))

        # Convert to OpenCV-compatible format
        image_for_shap = preprocessed_image[0]  # Shape: (128, 128, 3)

        # Initialize SHAP
        masker = shap.maskers.Image("blur(128,128)", image_for_shap.shape)
        explainer = shap.Explainer(app.state.cnn_model, masker, output_names=class_names.tolist())

        # Compute SHAP values
        shap_values = explainer(preprocessed_image)

        # Create the SHAP image plot
        plt.figure()
        shap.image_plot(shap_values, preprocessed_image, show=False)  # Static plot, no pop-up

        # Save the SHAP image to a BytesIO stream
        shap_image_path_in_memory = BytesIO()
        plt.savefig(shap_image_path_in_memory, format="png", bbox_inches='tight', pad_inches=0)
        shap_image_path_in_memory.seek(0)  # Rewind the BytesIO object to the beginning
        plt.close()  # Ensure no pop-up

        # Upload SHAP image to Google Cloud Storage
        shap_image_url = upload_to_gcs(shap_image_path_in_memory, bucket_name)

        return {"shap_image_url": shap_image_url}

    except Exception as e:
        print(f"Error during SHAP explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during SHAP explanation: {str(e)}")
