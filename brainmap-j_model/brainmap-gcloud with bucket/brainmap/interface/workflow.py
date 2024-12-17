import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from taxifare.interface.main import evaluate, preprocess, train
from taxifare.ml_logic.registry import mlflow_transition_model
from taxifare.params import *

@task
def preprocess_new_data(min_date: str, max_date: str):
    return preprocess(min_date=min_date, max_date=max_date)

@task
def evaluate_production_model(min_date: str, max_date: str):
    return evaluate(min_date=min_date, max_date=max_date)

@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    return train(min_date=min_date, max_date=max_date, split_ratio=split_ratio)

@task
def transition_model(current_stage: str, new_stage: str):
    return mlflow_transition_model(current_stage=current_stage, new_stage=new_stage)

@task
def notify(old_mae, new_mae):
    """
    Notify about the performance
    """
    base_url = 'https://chat.api.lewagon.com'
    channel = '802'
    url = f"{base_url}/{channel}/messages"
    author = 'krokrob'

    if new_mae < old_mae and new_mae < 2.5:
        content = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"
    elif old_mae < 2.5:
        content = f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    else:
        content = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"

    data = dict(author=author, content=content)

    response = requests.post(url, data=data)
    response.raise_for_status()


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """

    min_date = EVALUATION_START_DATE
    max_date = str(datetime.strptime(min_date, "%Y-%m-%d") + relativedelta(months=1)).split()[0]

    preprocessed = preprocess_new_data.submit(min_date=min_date, max_date=max_date)

    old_mae = evaluate_production_model.submit(min_date=min_date, max_date=max_date, wait_for=[preprocessed])
    new_mae = re_train.submit(min_date=min_date, max_date=max_date, split_ratio = 0.2, wait_for=[preprocessed])

    old_mae = old_mae.result()
    new_mae = new_mae.result()

    if new_mae < old_mae:
        print(f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}")
        transition_model.submit(current_stage="Staging", new_stage="Production")
    else:
        print(f"🚀 Old model kept in place with MAE: {old_mae}. The new MAE was: {new_mae}")

    notify.submit(old_mae, new_mae)

if __name__ == "__main__":
    train_flow()

################################################################
################################################################MARKUS' 5 CENTS
################################################################

import os
from prefect import task, flow

# Assuming the necessary functions are imported or implemented:
from mri_processing import load_mri_data, normalize_mri, one_hot_encode
from ml_logic import initialize_model, train_model, evaluate_model, predict

# Global constants (adjust these as necessary)
TRAIN_DATA_DIR = "../data/raw_data/Training"
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42

@task
def load_data(train_data_dir, validation_split, image_size, batch_size, seed):
    """
    Load and preprocess the MRI data.
    """
    return load_mri_data(train_data_dir, validation_split=VALIDATION_SPLIT, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED)

@task
def normalize_mri(dataset, label):
    return normalize_mri(dataset, label)

@task
def train_model(train_data, val_data):
    """
    Train a model on the given training and validation datasets.
    """

    return train(data_path: str = "data/processed", split_ratio: float = 0.2, epochs: int = 50) -> float:

@task
def evaluate_model(model_path, val_data):
    """
    Evaluate the model on the validation dataset.
    """
    return evaluate_model(...)

@task
def predict_image(model_path, image_path):
    """
    Predict the class of a user-uploaded MRI image using the model.
    """
    return model.predict(...)

@flow(name="MRI_Model_Flow")
def mri_fixed_model_flow(image_path=None):
    """
    Flow for training, evaluating, and optionally predicting with an MRI model.
    If `image_path` is provided, it predicts the class of the uploaded image.
    """
    # Load and preprocess the data
    train_data, val_data = load_and_preprocess_data.submit(
        train_data_dir=TRAIN_DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    # Train the fixed model
    fixed_model_path = train_fixed_model.submit(train_data=train_data, val_data=val_data)

    # Evaluate the fixed model
    metrics = evaluate_fixed_model.submit(model_path=fixed_model_path, val_data=val_data)

    print(f"🚀 Fixed model workflow complete. Final MAE: {metrics.result()['mae']}")

    # Predict on user-uploaded image if provided
    if image_path:
        prediction = predict_image.submit(model_path=fixed_model_path.result(), image_path=image_path)
        print(f"🔮 Prediction result: {prediction.result()}")

if __name__ == "__main__":
    # Run the workflow and optionally pass an image path for prediction
    user_image_path = "path/to/user/image.jpg"  # Replace with user-uploaded image path
    mri_fixed_model_flow(image_path=user_image_path)
