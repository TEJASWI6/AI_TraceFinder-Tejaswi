import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from skimage.feature import local_binary_pattern

# --- CONFIGURATION ---
IMG_SIZE = 256

# --- Helper Functions ---
@st.cache_data
def load_model():
    """Loads the final v2 saved model and label encoder."""
    try:
        model = joblib.load('final_model_v2.joblib')
        label_encoder = joblib.load('final_label_encoder_v2.joblib')
        return model, label_encoder
    except FileNotFoundError:
        return None, None

def extract_lbp_features(image_array):
    """Extracts LBP features from a numpy image array."""
    try:
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

# --- Main App ---
st.set_page_config(page_title="TraceFinder", layout="wide")
st.title("TraceFinder 🕵️‍♂️")
st.write("A forensic tool to identify a document's source or status (Original, Tampered, or Scanned).")

model, le = load_model()

if model is None or le is None:
    st.error("Model files not found! Please ensure 'final_model_v2.joblib' and 'final_label_encoder_v2.joblib' are in the GitHub repository.")
else:
    uploaded_file = st.file_uploader("Upload a document...", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"])

    if uploaded_file is not None:
        img_array = None
        
        if uploaded_file.type == "application/pdf":
            st.write("PDF detected, converting first page to image...")
            try:
                # This version for deployment does NOT need the poppler_path
                images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
                if images:
                    image = images[0]
                    img_array = np.array(image)
            except Exception as e:
                st.error(f"Error converting PDF. The server might be having trouble. Please try again. Error: {e}")
        else:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

        if img_array is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption='Uploaded Document', use_column_width=True)
            with col2:
                with st.spinner('Analyzing document...'):
                    features = extract_lbp_features(img_array)
                    if features is not None:
                        prediction_proba = model.predict_proba([features])[0]
                        confidence = np.max(prediction_proba)
                        predicted_class_index = np.argmax(prediction_proba)
                        prediction = le.inverse_transform([predicted_class_index])[0]
                        st.success('Analysis Complete!')
                        st.subheader("Result:")
                        if prediction == "Tampered":
                            st.error("Status: Tampered")
                            st.write("This document shows characteristics consistent with digital alteration.")
                        elif prediction == "Original":
                            st.info("Status: Original Document")
                            st.write("This appears to be an original, unaltered document, not a scan.")
                        else:
                            st.success("Status: Authentic Scan")
                            st.metric(label="Predicted Scanner Model", value=prediction, delta=f"{confidence*100:.2f}% Confidence")
                        with st.expander("View Detailed Probabilities"):
                            prob_df = pd.DataFrame(prediction_proba, index=le.classes_, columns=['Probability'])
                            st.bar_chart(prob_df)

