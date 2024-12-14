from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from interface.workflow import predict

app = FastAPI()

@app.post("/predict")
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Call the predict function
        result = predict(image)

        return {
            "Predicted Class": result["predicted_class"],
            "Probability": result["probability"],
            "SHAP Explanation": result["shap_image"],
            "YOLO Tumor Detection": result["yolo_image"],
        }
    except Exception as e:
        return {"error": str(e)}
