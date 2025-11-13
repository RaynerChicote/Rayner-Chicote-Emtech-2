import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os

# ------------------------------
# Constants
# ------------------------------
MODEL_URL = "https://drive.google.com/file/d/17VK-PaP62fJqtP2FvFmQL4eAazyz5C9R/view?usp=sharing"  # replace with your link
MODEL_LOCAL_PATH = "model.h5"

# ------------------------------
# Download and cache the model
# ------------------------------
@st.cache_resource
def load_model_from_url(url, local_path):
    if not os.path.exists(local_path):
        # download model if not exists
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(local_path)

model = load_model_from_url(MODEL_URL, MODEL_LOCAL_PATH)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Image Prediction App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # adjust target_size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = img_array / 255.0  # normalize if needed

    # Make prediction
    try:
        prediction = model.predict(img_array)[0][0]  # for single-output sigmoid
        st.write(f"Prediction (sigmoid output): {prediction:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
