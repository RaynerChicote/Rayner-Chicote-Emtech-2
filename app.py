import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
import gdown
import os

# ---------------------------------------------------------
# 🔗 Google Drive Model Download
# ---------------------------------------------------------
# Replace this with YOUR actual Google Drive file ID:
# (Example link: https://drive.google.com/file/d/1AbCdEfGhIJkl/view?usp=sharing)
# File ID is: 1AbCdEfGhIJkl
MODEL_FILE_ID = "17VK-PaP62fJqtP2FvFmQL4eAazyz5C9R"
MODEL_PATH = "tyre_quality_model.h5"

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ---------------------------------------------------------
# 🧠 Load Model (cached for speed)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------------------------------------------------
# 🏷️ Class Labels (update based on your dataset)
# ---------------------------------------------------------
# Example: ["bad", "good", "worn"]
CLASS_NAMES = ['defective', 'good']  # 🔧 change to your actual classes

# ---------------------------------------------------------
# 🌟 Streamlit App UI
# ---------------------------------------------------------
st.title("🛞 Tyre Quality Detection App")
st.write("Upload a tyre image and let the deep learning model predict its quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Tyre Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"### 🏁 Prediction: **{predicted_class.upper()}**")
    st.progress(float(confidence) / 100)
    st.write(f"Confidence: **{confidence:.2f}%**")

else:
    st.info("📸 Please upload a tyre image to begin.")

# ---------------------------------------------------------
# 🧾 Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by Rayner Chicote | Tyre Quality Classifier using Transfer Learning")
