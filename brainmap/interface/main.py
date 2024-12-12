import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

# Import your custom modules for MRI image analysis
from brainmap.params import *
from brainmap.data import load_mri_data, preprocess_mri_data, split_data
from brainmap.model import initialize_model, compile_model, train_model, evaluate_model
from brainmap.registry import load_model, save_model, save_results

def preprocess(data_path: str = "training_dir", output_path: str = "data/processed") -> None:

    print(Fore.MAGENTA + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)

    images, labels = load_mri_data(data_path)
    processed_images, processed_labels = preprocess_mri_data(images, labels)

    # Save processed data
    np.save(Path(output_path) / "images.npy", processed_images)
    np.save(Path(output_path) / "labels.npy", processed_labels)

    print("✅ Preprocessing complete. Processed data saved.\n")

def train(data_path: str = "data/processed", split_ratio: float = 0.2, epochs: int = 50) -> float:

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

    images = np.load(Path(data_path) / "images.npy")
    labels = np.load(Path(data_path) / "labels.npy")

    X_train, X_val, y_train, y_val = split_data(images, labels, split_ratio)

    model = initialize_model(input_shape=X_train.shape[1:], num_classes=len(np.unique(labels)))
    model = compile_model(model)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)

    # Save the trained model and results
    save_model(model)
    save_results(history.history)

    print("✅ Training complete. Model and results saved.\n")

def evaluate(data_path: str = "data/processed", stage: str = "Production") -> float:

    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    # Load preprocessed test data
    images = np.load(Path(data_path) / "images.npy")
    labels = np.load(Path(data_path) / "labels.npy")

    # Load trained model
    model = load_model(stage=stage)

    # Evaluate model performance
    metrics = evaluate_model(model, images, labels)

    print(f"✅ Evaluation complete. Metrics: {metrics}\n")

def predict(image_path: str) -> np.ndarray:
    """
    Predict using the trained MRI image classification model:
    - Load a single MRI scan.
    - Preprocess it and make a prediction using the trained model.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)

    # Load and preprocess the input image
    image = preprocess_mri_data([image_path], None)[0]

    # Load trained model
    model = load_model()

    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))

    print(f"✅ Prediction complete. Result: {prediction}\n")

if __name__ == "__main__":
    preprocess(data_path="data/raw_mri_images", output_path="data/processed")
    train(data_path="data/processed", split_ratio=0.2, epochs=50)
    evaluate(data_path="data/processed", stage="Production")
    predict("data/raw_mri_images/image1.jpg")
