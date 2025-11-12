import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
from PIL import Image
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Rice Leaf Disease Detector",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8ff;
        margin: 10px 0;
    }
    .disease-info {
        background-color: #fffacd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class information"""
    try:
        model = load_model('rice_leaf_model.h5')
        with open('class_info.json', 'r') as f:
            class_info = json.load(f)
        return model, class_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(img):
    """Preprocess the image for prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_disease_info(disease_name):
    """Get information about the detected disease"""
    disease_info = {
        "Brown Spot": {
            "symptoms": "Small, oval brown spots on leaves, may have yellow halo",
            "treatment": "Use resistant varieties, proper fertilization, fungicide application",
            "prevention": "Crop rotation, proper field sanitation"
        },
        "Leaf Blast": {
            "symptoms": "Diamond-shaped lesions with gray centers and brown borders",
            "treatment": "Fungicides like tricyclazole, proper water management",
            "prevention": "Use resistant varieties, avoid excessive nitrogen"
        },
        "Hispa": {
            "symptoms": "White streaks and patches on leaves caused by insect feeding",
            "treatment": "Insecticides, biological control with natural enemies",
            "prevention": "Field sanitation, remove infected leaves"
        },
        "Healthy": {
            "symptoms": "No visible symptoms, green and vibrant leaves",
            "treatment": "Maintain current practices",
            "prevention": "Regular monitoring, proper nutrition"
        }
    }
    return disease_info.get(disease_name, {
        "symptoms": "Information not available",
        "treatment": "Consult agricultural expert",
        "prevention": "Regular monitoring and proper cultivation practices"
    })

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Leaf Disease Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This AI-powered app helps identify common rice leaf diseases. "
        "Upload an image of a rice leaf to get instant diagnosis and treatment recommendations."
    )
    
    st.sidebar.title("Supported Diseases")
    st.sidebar.write("""
    - **Brown Spot**
    - **Leaf Blast** 
    - **Hispa**
    - **Healthy Leaves**
    """)
    
    st.sidebar.title("Instructions")
    st.sidebar.write("""
    1. Upload a clear image of a rice leaf
    2. Wait for the AI analysis
    3. View the diagnosis and recommendations
    4. Take appropriate action
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Rice Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a rice leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Uploaded Image", use_column_width=True)
            
            # Process the image
            processed_image = preprocess_image(image_display)
            
            # Load model (cached)
            model, class_info = load_model_and_classes()
            
            if model and class_info:
                # Make prediction
                with st.spinner('Analyzing the image...'):
                    predictions = model.predict(processed_image)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                    
                    class_names = class_info['class_names']
                    predicted_class = class_names[predicted_class_idx]
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Prediction box
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("🔍 Diagnosis Result")
                    st.write(f"**Detected Condition:** {predicted_class}")
                    st.write(f"**Confidence Level:** {confidence:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Disease information
                    disease_data = get_disease_info(predicted_class)
                    st.markdown('<div class="disease-info">', unsafe_allow_html=True)
                    st.subheader("📋 Disease Information")
                    st.write(f"**Symptoms:** {disease_data['symptoms']}")
                    st.write(f"**Treatment:** {disease_data['treatment']}")
                    st.write(f"**Prevention:** {disease_data['prevention']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence scores for all classes
                    st.subheader("📊 Confidence Scores")
                    for i, (class_name, score) in enumerate(zip(class_names, predictions[0])):
                        progress_bar = st.progress(float(score))
                        st.write(f"{class_name}: {score:.2%}")
                    
            else:
                st.error("Model could not be loaded. Please check if the model files are available.")
    
    with col2:
        st.subheader("Example Images")
        st.write("For best results, upload images similar to these:")
        
        # Create example images section (you can replace these with actual example images)
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.write("**Clear leaf image**")
            st.info("• Good lighting\n• Single leaf\n• Clear focus")
        
        with example_col2:
            st.write("**What to avoid**")
            st.error("• Blurry images\n• Multiple leaves\n• Poor lighting")
        
        st.subheader("💡 Prevention Tips")
        st.write("""
        - Regularly monitor your rice field
        - Maintain proper water management
        - Use resistant varieties when available
        - Practice crop rotation
        - Ensure proper fertilization
        - Remove infected plants promptly
        """)

if __name__ == "__main__":
    main()
