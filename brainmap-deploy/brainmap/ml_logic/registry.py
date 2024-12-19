from tensorflow import keras
from google.cloud import storage
import os
import time
import pickle
from brainmap.params import *

def load_model(model_path="../api/checkpoints/best_model.keras") -> keras.Model:
    """
    Load the model from the specified path.
    """
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No such file or directory: '{model_path}'")

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise OSError(f"Unable to load model from '{model_path}': {e}")

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.keras" --> unit 02 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")
    model.save(model_path)

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        return None
    return None

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(LOCAL_REGISTRY_PATH, "params", f"{timestamp}.pickle")

    with open(results_path, "wb") as f:
        pickle.dump({"params": params, "metrics": metrics}, f)

    print(f"Results saved to {results_path}")
