import streamlit as st
import requests
from PIL import Image
from io import BytesIO


# Set FastAPI base URL
BASE_URL = "https://brainmapimage-795064670774.europe-west1.run.app"


# Title of the app
st.set_page_config(page_title="Brainmap, MRI meets AI", layout="wide")

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
    <h1 class="main-title"> Advanced MRI Analysis for Tumor Detections</h1>
    """,
    unsafe_allow_html=True
)


logo = Image.open("./static/images/logo.png")
st.sidebar.image(logo, width=300)
options = ["Predict (CNN)", "Predict (YOLO)", "Explainable AI", "All Results"]
choice = st.sidebar.radio("Choose an operation", options)
st.sidebar.markdown("---")
st.sidebar.write('CONTACT US')
st.sidebar.write('Joël Tiogo, Eng, MBA')
st.sidebar.write(':envelope: mailto:tiogojoel@gmail.com')
st.sidebar.write('https://www.linkedin.com/in/joeltiogo/')

st.sidebar.write('Gökhan Dede, PhD')
st.sidebar.write(':envelope: mailto:gokhandede.ai@gmail.com')
st.sidebar.write('https://www.linkedin.com/in/gokhan-dede/')

st.sidebar.write('Antonio Ferri, CFA')
st.sidebar.write(':envelope: mailto:antonio.ferri@gmail.com')
st.sidebar.write('https://www.linkedin.com/in/ferriantonio/')

st.sidebar.write('Markus Kaller, PhD')
st.sidebar.write(':envelope: mailto:markus_kaller@gmx.de')
st.sidebar.write('https://www.linkedin.com/in/markus-kaller/')

st.sidebar.write('Mosleh Rostami, MSc')
st.sidebar.write(':envelope: mailto:Mosleh.rostami@gmail.com')
st.sidebar.write('https://www.linkedin.com/in/mosleh-rostami-50aa6238/')

st.sidebar.markdown("---")
st.sidebar.write("[GitHub](https://github.com/joelbrice/brainmap)")
st.sidebar.write("[documentation](https://joelbrice.github.io/brainmap.github.io/)")

st.markdown("""
### Welcome to **BrainMap**!

**BrainMap** is a cutting-edge platform leveraging artificial intelligence to revolutionize brain tumor detection and classification. Our state-of-the-art deep learning algorithms analyze MRI scans with precision, providing rapid and accurate assessments to support medical professionals in their diagnostic process.

#### How It Works:
- **Upload an MRI Image**: Simply upload your MRI image using the interface.
- **Get Instant Predictions**: Our advanced system will swiftly process the image and deliver a comprehensive analysis indicating the presence or absence of tumors.

#### Why Choose BrainMap?
- **Reliable**: Combining the power of machine learning with medical expertise.
- **Efficient**: Non-invasive tool for early tumor detection.
- **Improving Outcomes**: Potentially enhances patient outcomes through timely intervention.

Thank you for choosing BrainMap as your trusted partner in medical imaging analysis!
""")
# File upload
st.write("### 📂 Upload an MRI Image")
uploaded_file = st.file_uploader("Supported formats: JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Uploaded MR Image**")
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
            st.write("**📊 Probability:**", f"{result['Probability']*100}%")
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
                st.write("**🔍Tumor Detected:**")
            for detection in result.get("yolo_detections", []):
                st.write(f"- **Class:** {detection['class_name']}, **Confidence:** {detection['confidence']*100}%")

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

    elif choice == "Explainable AI":
        st.header("📖 Understanding how AI analyses the MRI using explainable AI")

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
            st.write("**📊 Probability:**", f"{result_cnn['Probability']*100}%")
        else:
            st.error(f"CNN Error: {response_cnn.json().get('detail', 'Unknown error')}")


        # YOLO Detection
        with st.spinner("Fetching YOLO Detection..."):
            files = {"file": uploaded_file}
            response_yolo = requests.post(f"{BASE_URL}/predict_yolo/", files=files)

        if response_yolo.status_code == 200:
            result_yolo = result_yolo.json()
            with col2:
                st.write("**🔍 Detected Tumors:**")
            for detection in result_yolo.get("yolo_detections", []):
                st.write(f"- **Class:** {detection['class_name']}, **Confidence:** {detection['confidence']*100}%")

            yolo_image_url = result_yolo.get('yolo_image_url')
            if yolo_image_url:
                response_image = requests.get(yolo_image_url)
                if response_image.status_code == 200):
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

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2C3E50;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
    }
    </style>
    <div class="footer">
        © 2024 BrainMap. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
