from workflow import predict

def main(image_path: str):
    """
    Main function to handle the prediction process.
    Calls the predict function from workflow to get results.
    """
    result = predict(image_path)

    # Output results
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Prediction Probability: {result['probability']}")
    print(f"SHAP Explanation Image saved at: {result['shap_image']}")
    print(f"YOLO Detection Image saved at: {result['yolo_image']}")

if __name__ == "__main__":
    image_path = "../../data/raw_data/Testing/glioma/Tel-gl_0010.jpg"
    main(image_path)
