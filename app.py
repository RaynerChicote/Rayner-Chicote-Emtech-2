import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json, os, gdown

st.title("🌾 Rice Leaf Disease Detection App")
st.write("Upload a rice leaf image to detect its condition.")

# --- Google Drive model setup ---
model_file = "rice_leaf_best_model.h5"  # or .keras if you saved in that format
file_id = "1Sz-aFfWFujWSiGH2Iu1wsDpTvvsYAHtN"
drive_url = f"https://drive.google.com/uc?id={file_id}"

# --- Download model from Google Drive if not found ---
if not os.path.exists(model_file):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(drive_url, model_file, quiet=False)

# --- Verify model file integrity ---
if not os.path.exists(model_file) or os.path.getsize(model_file) < 1000:
    st.error("❌ Model download failed. Please check your Google Drive link or permissions.")
    st.stop()

# --- Load Model ---
try:
    model = tf.keras.models.load_model(model_file)
    st.success(" Model loaded successfully!")
except Exception as e:
    st.error(f" Error loading model: {e}")
    st.stop()

# --- Load class names ---
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except:
    st.warning(" class_names.json not found, using placeholder names.")
    class_names = [f"Class {i}" for i in range(model.output_shape[-1])]

# --- File uploader for prediction ---
uploaded_file = st.file_uploader("Upload a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing image...")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
