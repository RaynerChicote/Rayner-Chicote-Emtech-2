import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO

# ------------------------------
# Constants
# ------------------------------
MODEL_URL = "YOUR_H5_MODEL_URL_HERE"  # replace with your link

# ------------------------------
# Load model with caching
# ------------------------------
@st.cache_resource
def load_model_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # raise error if download fails
    model_file = BytesIO(response.content)
    return tf.keras.models.load_model(model_file)

model = load_model_from_url(MODEL_URL)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Image Prediction App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # adjust target_size to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # make batch dimension
    img_array = img_array / 255.0  # normalize if needed

    # ------------------------------
    # Make prediction
    # ------------------------------
    try:
        prediction = model.predict(img_array)[0][0]  # single value for sigmoid
        st.write(f"Prediction (sigmoid output): {prediction:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
