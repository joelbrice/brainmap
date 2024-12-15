
import streamlit as st
import requests
import time
from PIL import Image
import numpy as np
from brainmap.params import *


st.markdown('''
<style>
.big-font {
    font-size:20px !important;
}
</style>
''', unsafe_allow_html=True)

'''
# brainmap, a tumor detection platform
'''

st.markdown('<p class="big-font">Welcome to Brainmap, a platform for tumor detection. Upload an MRI image and get a prediction on whether it indicates the presence of tumor or not.</p>', unsafe_allow_html=True)


'''
## Upload MRI Image
'''

# URL of the API
url = 'https://' # SERVICE_URL


# Placeholder for the image
image_placeholder = st.empty()

# Upload_file:
uploaded_file = st.file_uploader("Choose an MRI image...", type=["tif", "tiff", "png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, width=300, caption="Uploaded image")
    if st.button('Get Prediction and the Probability of having a tumor'):
        # Loading bar starts here
        with st.spinner('Processing the image...'):
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)

        # Proceed with the image processing and prediction
        img = uploaded_file.getvalue()
        files = {"file": ("image.jpeg", img, "image/jpeg")}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            if "Prediction" in result and 'Probability' in result:
                prediction = result["Prediction"]
                probability = result["Probability"]
                if prediction is not None:
                    pred = prediction * 100
                    prob = probability * 100
                    if prediction < 0.5:
                        pred = 100 - pred
                        st.markdown(f"<h2 style='color:red;'>Tumor is detected</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color:black;'>Prediction: {pred:.2f}%</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color:black;'>Probability: {prob:.2f}%</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color:green;'>No tumor detected</h2>", unsafe_allow_html=True)
                        #st.markdown(f"<h3 style='color:black;'>Prediction: {pred:.2f}%</h3>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color:black;'>Probability: {prob:.2f}%</h3>", unsafe_allow_html=True)
                else:
                    st.error("Error: Prediction value is None.")
            else:
                st.error("Error: 'Prediction' key not found in API response.")
        else:
            st.error(f"Error in API request. Status code: {response.status_code}")

# Footer
st.markdown('<p class="big-font">Thank you for using brainmap!</p>', unsafe_allow_html=True)
