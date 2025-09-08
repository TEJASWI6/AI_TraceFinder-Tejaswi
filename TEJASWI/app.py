import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import local_binary_pattern

# --- CONFIGURATION ---
IMG_SIZE = 256

# --- Helper Functions ---
# Use st.cache_data to load the model only once
@st.cache_data
def load_model():
    """Loads the final saved model and label encoder."""
    try:
        model = joblib.load('finale_model.joblib')
        label_encoder = joblib.load('finale_label_encoder.joblib')
        return model, label_encoder
    except FileNotFoundError:
        return None, None

def extract_lbp_features(image_array):
    """Extracts LBP features from a numpy image array."""
    try:
        # Convert to grayscale if it's a color image
        if len(image_array.shape) == 3:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image_array
            
        img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_resized, n_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

# --- Main Application UI ---
st.set_page_config(page_title="TraceFinder", layout="wide")
st.title("TraceFinder 🕵️‍♂️")
st.write("A forensic tool to identify a document's source scanner or detect tampering.")

# Load the trained model and label encoder
model, le = load_model()

if model is None or le is None:
    st.error("Model files not found! Please make sure 'final_model.joblib' and 'final_label_encoder.joblib' are in the app folder.")
else:
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload a scanned document...", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Document', use_column_width=True)
        
        with col2:
            # Show a spinner while processing
            with st.spinner('Analyzing document...'):
                img_array = np.array(image)
                features = extract_lbp_features(img_array)

                if features is not None:
                    # Make prediction
                    prediction_proba = model.predict_proba([features])[0]
                    confidence = np.max(prediction_proba)
                    predicted_class_index = np.argmax(prediction_proba)
                    prediction = le.inverse_transform([predicted_class_index])[0]
                    
                    st.success('Analysis Complete!')
                    st.subheader("Result:")

                    # This is the final logic you designed
                    if prediction == "Tampered":
                        st.error("Status: Tampered")
                        st.write("This document shows characteristics consistent with digital alteration.")
                    else:
                        st.success("Status: Authentic")
                        st.metric(label="Predicted Scanner Model", value=prediction, delta=f"{confidence*100:.2f}% Confidence")

                    # Show a detailed breakdown of probabilities
                    with st.expander("View Detailed Probabilities"):
                        prob_df = pd.DataFrame(prediction_proba, index=le.classes_, columns=['Probability'])
                        st.bar_chart(prob_df)
