import os
import numpy as np
from pathlib import Path

from colorama import Fore, Style

from brainmap.ml_logic.data import load_mri_data
# from brainmap.ml_logic.model import evaluate_model
from tensorflow.keras.models import load_model

checkpoint_dir = './checkpoints'
tensorboard_log_dir = './logs'

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)
# Load the best model
cnn_model_path = os.path.join(checkpoint_dir, 'best_model.keras')
cnn_model_path

cnn_model = load_model(cnn_model_path)

# Evaluate on test data
test_loss, test_accuracy, test_auc, test_precision, test_recall = cnn_model.evaluate(X_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")


import requests

url = "http://127.0.0.1:8000/predict"
files = {'file': open('/home/joelbrice/code/joelbrice/brainmap/data/raw_data/Testing/glioma/Te-gl_0010.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())

def preprocess(data_path: str = "../../data/raw_data", output_path: str = "../../data/processed") -> None:
    print(Fore.MAGENTA + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Load and preprocess MRI data
    processed__train_img, processed_val_img = load_mri_data(data_path)


    # Save processed data
    np.save(Path(output_path) / "images.npy", processed__train_img)
    np.save(Path(output_path) / "labels.npy", processed_val_img)

    print("✅ Preprocessing complete. Processed data saved.\n")

def train(data_path: str = "../../data/processed", split_ratio: float = 0.2, epochs: int = 50) -> float:
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

    # Load preprocessed data
    images = np.load(Path(data_path) / "images.npy")
    labels = np.load(Path(data_path) / "labels.npy")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(images, labels, split_ratio)

    # Initialize and compile model
    model = initialize_model(input_shape=X_train.shape[1:], num_classes=len(np.unique(labels)))
    model = compile_model(model)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)

    # Save the trained model and results
    save_model(model)
    save_results(history.history)

    print("✅ Training complete. Model and results saved.\n")

def evaluate(data_path: str = "../../data/processed") -> float:
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    # Load preprocessed test data
    images = np.load(Path(data_path) / "images.npy")
    labels = np.load(Path(data_path) / "labels.npy")

    # Load trained model
    model = load_model()

    # Evaluate model performance
    metrics = evaluate_model(model, images, labels)

    print(f"✅ Evaluation complete. Metrics: {metrics}\n")
    return metrics

def predict(image_path: str) -> None:
    """
    Predict using the trained MRI image classification model:
      - Load a single MRI scan.
      - Preprocess it and make a prediction using the trained model.
      - Output probabilities and predicted class.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)

    # Load and preprocess the input image
    image = load_mri_data(image_path)
    #preprocessed_image = preprocess(image, image_path)
    # Load trained model
    model = load_model()

    if not model:
        raise ValueError("No trained model found. Please train the model first.")

    # Make prediction
    prediction_probabilities = model.predict(np.expand_dims(image, axis=0))

    # Get predicted class with highest probability
    predicted_class_index = np.argmax(prediction_probabilities)

    # Assuming `class_names` contains class labels (e.g., ["glioma", "meningioma", "pituitary tumor"])
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    predicted_class_label = class_names[predicted_class_index]

    print('✅ Prediction complete.')

    print(f"Predicted Class: {predicted_class_label}")

    print(f"Prediction Probabilities: {prediction_probabilities}")

if __name__ == "__main__":
    #preprocess(data_path="../../data/Training/", output_path="../../data/processed")

    # Uncomment these lines as needed:
    # train(data_path="../../data/processed", split_ratio=0.2, epochs=100)
    # evaluate(data_path="../../data/Testing", stage="Production")
    predict("../../data/raw_data/Testing/glioma/Tel-gl_0010.jpg")
