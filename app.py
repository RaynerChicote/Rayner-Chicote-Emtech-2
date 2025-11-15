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
# 🏷️ Class Labels
# ---------------------------------------------------------
CLASS_NAMES = ['defective', 'good']

# ---------------------------------------------------------
# 🌟 Streamlit App UI
# ---------------------------------------------------------
st.title("🛞 Tyre Quality Detection App")
st.write("Upload a tyre image and let the deep learning model predict its quality.")

# ---------------------------------------------------------
# 📚 Tyre Quality Information (Now Always Visible)
# ---------------------------------------------------------
st.markdown("## 🔍 Tyre Quality Information")
st.write("""
This model predicts the quality of a tyre based on its image, classifying it as either **Good** or **Defective**.  
Below is some important information on both categories:

---

### ✅ **Good Tyre Quality**
A **good** tyre is in proper condition for safe driving.

**Key Characteristics of Good Tyres:**
- Proper tread depth (provides adequate grip).
- Even wear with no significant damage.
- No visible cracks, cuts, or punctures.

**Maintenance Tips:**
- Keep tyres inflated to the correct pressure.
- Regularly check tread depth and inspect for any defects.
- Rotate tyres every 6,000–8,000 miles.

---

### ⚠️ **Defective Tyre Quality**
A **defective** tyre poses a safety risk and must be replaced immediately.

**Common Defects:**
- Bald or worn-out tread.
- Bulges in the sidewall.
- Cuts, punctures, or cracks in the rubber.

**Immediate Action:**
- Replace the defective tyre immediately.
- Do not drive on defective tyres.
- Perform routine tyre inspections for safety.

---
""")

# ---------------------------------------------------------
# 📸 File uploader
# ---------------------------------------------------------
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

    # Add Tyre Quality Information based on the Prediction
    if predicted_class == 'good':
        st.markdown("### ✅ Good Tyre Quality")
        st.write("""
            A **good** tyre indicates proper condition for safe use.
            
            **Characteristics:**
            - Adequate tread depth  
            - No punctures or bulges  
            - Even wear  
            - No visible cracks  

            **Maintenance Tips:**
            - Check tyre pressure often  
            - Rotate tyres for even wear  
            - Inspect tyres regularly  
        """)
    else:
        st.markdown("### ⚠️ Defective Tyre Quality")
        st.write("""
            A **defective** tyre may have serious issues like cuts, bulges, or worn-out tread.
            
            **Common Defects:**
            - Bald tyres  
            - Sidewall bulges  
            - Punctures or cuts  
            - Rubber cracks  

            **Immediate Action:**
            - Replace the tyre immediately  
            - Avoid driving on defective tyres  
            - Inspect all tyres routinely  
        """)
else:
    st.info("📸 Please upload a tyre image to begin.")

# ---------------------------------------------------------
# 🧾 Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by Rayner Chicote | Tyre Quality Classifier using Transfer Learning")
