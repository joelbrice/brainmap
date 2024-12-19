
from brainmap.ml_logic.registry import load_model


def test_load_model():
    model_path = "../brainmap/api/checkpoints/best_model.keras"  # Adjust the path as needed
    model = load_model(model_path)
    print("Model summary:")
    model.summary()

if __name__ == "__main__":
    test_load_model()
