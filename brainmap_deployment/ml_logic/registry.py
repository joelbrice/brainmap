import os
import logging
from tensorflow import keras
from google.cloud import storage
from ml_logic.params import BUCKET_NAME, CNN_MODEL_PATH, YOLO_MODEL_PATH

# Setting up logging
logging.basicConfig(level=logging.INFO)

def download_model_from_gcs(model_path: str, bucket_name: str) -> str:
    """
    Download a model from Google Cloud Storage (GCS) to a temporary local path.

    Args:
        model_path (str): Path to the model in GCS.
        bucket_name (str): Name of the GCS bucket.

    Returns:
        str: Local path to the downloaded model file.
    """
    # Extract model filename
    model_filename = os.path.basename(model_path)
    local_model_path = f"/tmp/{model_filename}"

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)

    logging.info(f"Downloading model from GCS: {bucket_name}/{model_path} to {local_model_path}...")
    blob.download_to_filename(local_model_path)
    logging.info(f"✅ Model downloaded successfully from {bucket_name}/{model_path}!")

    return local_model_path

def load_cnn_model() -> keras.Model:
    """
    Load the CNN model from GCS and return it.

    Returns:
        keras.Model: Loaded CNN model.
    """
    local_cnn_path = download_model_from_gcs(CNN_MODEL_PATH, BUCKET_NAME)
    logging.info("Loading CNN model from local path...")
    cnn_model = keras.models.load_model(local_cnn_path, compile=False)
    logging.info("✅ CNN model loaded successfully!")
    return cnn_model

def load_yolo_model():
    """
    Load the YOLO model from GCS and return it.

    Returns:
        YOLO: Loaded YOLO model.
    """
    from ultralytics import YOLO
    local_yolo_path = download_model_from_gcs(YOLO_MODEL_PATH, BUCKET_NAME)
    logging.info("Loading YOLO model from local path...")
    yolo_model = YOLO(local_yolo_path)
    logging.info("✅ YOLO model loaded successfully!")
    return yolo_model
