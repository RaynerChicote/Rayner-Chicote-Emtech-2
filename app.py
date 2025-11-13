import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Load the model
MODEL_PATH = "tyre_quality_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = ['bad', 'good', 'worn']  # Update based on your dataset folders

st.title("🛞 Tyre Quality Detection App")
st.write("Upload a tyre image to predict its quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class.upper()}** ({confidence:.2f}% confidence)")
