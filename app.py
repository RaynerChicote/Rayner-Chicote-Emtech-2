import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# App title and intro
st.title("🌾 Rice Leaf Disease Detection App")
st.write("Upload an image of a rice leaf to detect if it is healthy or diseased.")

# Load model and class names
model = tf.keras.models.load_model("rice_leaf_best_model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Upload image
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

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

import gdown, os

model_path = "rice_leaf_best_model.h5"

# Download model from Google Drive if not found
if not os.path.exists(model_path):
    file_id = "1AbCDEFGHIJKLmnOPq"  # <-- put your Drive file ID here
    gdown.download(f"https://drive.google.com/uc?id=1aSs0w57K4_QL-K0uVxSmQASj3HlcHQzM", model_path, quiet=False)

import tensorflow as tf
model = tf.keras.models.load_model(model_path)
