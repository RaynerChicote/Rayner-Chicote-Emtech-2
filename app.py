import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# ------------------------------
# Google Drive download
# ------------------------------
MODEL_FILE_ID = "17VK-PaP62fJqtP2FvFmQL4eAazyz5C9R"  # replace with your .h5 ID
MODEL_PATH = "tyre_quality_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ------------------------------
# Class labels
# ------------------------------
CLASS_NAMES = ['defective', 'good']

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🛞 Tyre Quality Detector")
st.write("Upload a tyre image and the model will predict if it is defective or good.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]  # single value for sigmoid
    if prediction > 0.5:
        predicted_class = 'good'
        confidence = prediction * 100
    else:
        predicted_class = 'defective'
        confidence = (1 - prediction) * 100

    st.success(f"### Prediction: {predicted_class.upper()}")
    st.write(f"Confidence: {confidence:.2f}%")
else:
    st.info("📸 Please upload a tyre image to get started.")

st.markdown("---")
st.caption("Developed by [Your Name] | Tyre Quality Classifier using Transfer Learning")
