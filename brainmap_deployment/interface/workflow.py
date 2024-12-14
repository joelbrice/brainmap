import numpy as np
import tensorflow as tf
import shap
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO
from ml_logic.params import CNN_MODEL_PATH, YOLO_MODEL_PATH, CLASS_MAPPING, IMAGE_SIZE
from ml_logic.preprocessor import normalize_mri


def load_model(model_path, loader_func):
    """Generic function to load models."""
    return loader_func(model_path)


def preprocess_image(image):
    """Preprocess an input image for CNN."""
    image_resized = image.resize(IMAGE_SIZE)
    image_array = np.array(image_resized)
    normalized_image = normalize_mri(image_array)
    return np.expand_dims(normalized_image, axis=0)


def predict(image):
    """Main prediction logic for CNN and YOLO models."""
    # Preprocess the image
    image_array = preprocess_image(image)

    # Load CNN model
    cnn_model = load_model(CNN_MODEL_PATH, tf.keras.models.load_model)
    cnn_prediction = cnn_model.predict(image_array)
    predicted_class_index = np.argmax(cnn_prediction)
    predicted_class = list(CLASS_MAPPING.values())[predicted_class_index]
    probability = cnn_prediction[0][predicted_class_index]

    # Load YOLO model
    yolo_model = load_model(YOLO_MODEL_PATH, YOLO)
    tumor_detection = yolo_model.predict(source=image)
    detected_image = tumor_detection[0].plot()

    # SHAP explanation
    masker = shap.maskers.Image("blur(128,128)", image_array.shape[1:])
    explainer = shap.Explainer(cnn_model, masker, output_names=list(CLASS_MAPPING.values()))
    shap_values = explainer(image_array)

    # Save SHAP and YOLO images
    shap_image_path = './shap_image.png'
    plt.imshow(shap_values[0].values[0], cmap='coolwarm')
    plt.axis('off')
    plt.savefig(shap_image_path)

    yolo_image_path = './yolo_detection.png'
    plt.figure(figsize=(8, 8))
    plt.imshow(detected_image)
    plt.axis('off')
    plt.savefig(yolo_image_path)

    return {
        "predicted_class": predicted_class,
        "probability": probability,
        "shap_image": shap_image_path,
        "yolo_image": yolo_image_path,
    }
