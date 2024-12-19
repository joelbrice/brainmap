import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Set FastAPI base URL
BASE_URL = "http://127.0.0.1:8000"


# Title of the app
st.set_page_config(page_title="BrainMap: Tumor Detection", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495E;
    }
    .image-caption {
        font-size: 1rem;
        font-style: italic;
        text-align: center;
        color: #7F8C8D;
    }
    </style>
    <h1 class="main-title">🧠 BrainMap: Tumor Detection and Analysis</h1>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.markdown('<h2 class="sidebar-title">🔍 Navigation</h2>', unsafe_allow_html=True)
options = ["Predict (CNN)", "Predict (YOLO)", "Explain (CNN)", "All Results"]
choice = st.sidebar.radio("Choose an operation", options)

# File upload
st.sidebar.write("### 📂 Upload an MRI Image")
uploaded_file = st.sidebar.file_uploader("Supported formats: JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Uploaded MR Image:**")
        st.image(uploaded_file, width=450)

    if choice == "Predict (CNN)":
        st.header("📊 Tumor Prediction using CNN")

        # Send the image to the predict endpoint
        with st.spinner("Processing... Please wait."):
            response = requests.post(
                f"{BASE_URL}/predict",
                files={"file": uploaded_file.getvalue()}
            )

        if response.status_code == 200:
            result = response.json()
            st.write("**🧾 Prediction:**", result["Prediction"])
            st.write("**📊 Probability:**", f"{result['Probability']:.2f}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    elif choice == "Predict (YOLO)":
        st.header("🎯 Tumor Detection using YOLO")

        # Send the image to the YOLO endpoint
        with st.spinner("Processing... Please wait."):
            files = {"file": uploaded_file}
            response = requests.post(f"{BASE_URL}/predict_yolo/", files=files)

        if response.status_code == 200:
            result = response.json()
            with col2:
                st.write("**🔍 Detected Tumors:**")
            for detection in result.get("yolo_detections", []):
                st.write(f"- **Class:** {detection['class_name']}, **Confidence:** {detection['confidence']:.2f}")

            # Display the YOLO-detected image
            yolo_image_url = result.get('yolo_image_url')
            if yolo_image_url:
                response_image = requests.get(yolo_image_url)
                if response_image.status_code == 200:
                    yolo_image = Image.open(BytesIO(response_image.content))
                    with col2:
                        st.image(yolo_image, caption="Brain Tumor Detection", width=450)
                else:
                    st.error("Failed to retrieve YOLO detected image.")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    elif choice == "Explain (CNN)":
        st.header("📖 Explain CNN Prediction using SHAP")

        # Send the image to the explain endpoint
        with st.spinner("Processing... Please wait."):
            response = requests.post(
                f"{BASE_URL}/explain_cnn/",
                files={"file": uploaded_file.getvalue()}
            )

        if response.status_code == 200:
            result = response.json()

            # Display the SHAP explanation image
            shap_image_url = result.get('shap_image_url')
            if shap_image_url:
                response_image = requests.get(shap_image_url)
                if response_image.status_code == 200:
                    shap_image = Image.open(BytesIO(response_image.content))
                    st.image(shap_image, caption="SHAP Explanation", use_column_width=True)
                else:
                    st.error("Failed to retrieve SHAP explanation image.")
            else:
                st.error("No SHAP explanation image URL returned.")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")



    elif choice == "All Results":
        st.header("🌟 Combined Results")

        # CNN Prediction
        with st.spinner("Fetching CNN Prediction..."):
            response_cnn = requests.post(
                f"{BASE_URL}/predict",
                files={"file": uploaded_file.getvalue()}
            )

        if response_cnn.status_code == 200:
            result_cnn = response_cnn.json()
            st.write("**🧾 Prediction:**", result_cnn["Prediction"])
            st.write("**📊 Probability:**", f"{result_cnn['Probability']:.2f}")
        else:
            st.error(f"CNN Error: {response_cnn.json().get('detail', 'Unknown error')}")


        # YOLO Detection
        with st.spinner("Fetching YOLO Detection..."):
            files = {"file": uploaded_file}
            response_yolo = requests.post(f"{BASE_URL}/predict_yolo/", files=files)

        if response_yolo.status_code == 200:
            result_yolo = response_yolo.json()
            with col2:
                st.write("**🔍 Detected Tumors:**")
            for detection in result_yolo.get("yolo_detections", []):
                st.write(f"- **Class:** {detection['class_name']}, **Confidence:** {detection['confidence']:.2f}")

            yolo_image_url = result_yolo.get('yolo_image_url')
            if yolo_image_url:
                response_image = requests.get(yolo_image_url)
                if response_image.status_code == 200:
                    yolo_image = Image.open(BytesIO(response_image.content))
                    with col2:
                        st.image(yolo_image, caption="Brain Tumor Detection", width=450)
                else:
                    st.error("Failed to retrieve YOLO detected image.")
        else:
            st.error(f"YOLO Error: {response_yolo.json().get('detail', 'Unknown error')}")

        # SHAP Explanation
        with st.spinner("Fetching SHAP Explanation..."):
            response_shap = requests.post(
                f"{BASE_URL}/explain_cnn/",
                files={"file": uploaded_file.getvalue()}
            )

        if response_shap.status_code == 200:
            result_shap = response_shap.json()
            shap_image_url = result_shap.get('shap_image_url')
            if shap_image_url:
                response_image = requests.get(shap_image_url)
                if response_image.status_code == 200:
                    shap_image = Image.open(BytesIO(response_image.content))
                    st.image(shap_image, caption="SHAP Explanation", use_column_width=True)
                else:
                    st.error("Failed to retrieve SHAP explanation image.")
            else:
                st.error("No SHAP explanation image URL returned.")
        else:
            st.error(f"SHAP Error: {response_shap.json().get('detail', 'Unknown error')}")

else:
    st.info("Please upload an image to proceed.")
