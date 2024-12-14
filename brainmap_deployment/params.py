import os

# Class mappings for CNN predictions
CLASS_MAPPING = {
    "gl": "Glioma",
    "me": "Meningioma",
    "no": "No Tumor",
    "pi": "Pituitary",
}

# Model paths
CNN_MODEL_PATH = "models/cnn_model.h5"  # Example path in the GCS bucket
YOLO_MODEL_PATH = "models/yolo_model.pt"  # Example path in the GCS bucket

# GCS Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME", "brainmap")
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION", "europe-west1")

# Image processing parameters
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 1  # Processing one image at a time



GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")


# URL for deployment service (e.g., Streamlit or FastAPI)
SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8501")
