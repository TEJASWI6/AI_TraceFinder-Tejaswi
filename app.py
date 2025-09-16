import os
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from skimage.feature import local_binary_pattern

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide")

# --- CONFIGURATION ---
MODEL_PATH = 'baseline_model_v2.joblib'
ENCODER_PATH = 'baseline_label_encoder_v2.joblib'
LOG_FILE = 'prediction_log.csv'
IMG_SIZE = 256

# --- Feature Extraction Function ---
def extract_lbp_features(image):
    try:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_resized, n_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

# --- Load Model and Encoder ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        return model, label_encoder
    except FileNotFoundError:
        st.error(f"Model or encoder file not found. Make sure '{MODEL_PATH}' and '{ENCODER_PATH}' are in the same folder as app.py.")
        return None, None

model, le = load_model()

# --- Logging Function ---
def log_prediction(filename, prediction, confidence):
    new_log = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Filename': [filename],
        'Prediction': [prediction],
        'Confidence': [f"{confidence:.2f}%"]
    })
    
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = pd.concat([log_df, new_log], ignore_index=True)
    else:
        log_df = new_log
    
    log_df.to_csv(LOG_FILE, index=False)


# --- STREAMLIT APP INTERFACE ---
st.title("TraceFinder: Forensic Scanner Identification 🕵️‍♂️")
st.write("Upload a scanned document (Image or PDF) to identify its source or determine if it's an original/tampered file.")

# --- Sidebar for Logs ---
st.sidebar.title("History")
st.sidebar.write("View and download the complete prediction log.")
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.sidebar.dataframe(log_df)
    st.sidebar.download_button(
        label="Download Full Log (.csv)",
        data=log_df.to_csv(index=False).encode('utf-8'),
        file_name="full_prediction_log.csv",
        mime='text/csv'
    )
else:
    st.sidebar.info("No predictions have been logged yet.")


# --- Main Page Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file...", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"])

if uploaded_file is not None and model is not None:
    image_to_process = None
    if uploaded_file.type == "application/pdf":
        try:
            images = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=1)
            if images:
                image_to_process = images[0]
        except Exception as e:
            st.error(f"Error converting PDF: {e}")
    else:
        image_to_process = Image.open(uploaded_file)

    if image_to_process is not None:
        with col1:
            st.image(image_to_process, caption="Image being analyzed.", use_column_width=True)
        with col2:
            st.header("Analysis Result")
            with st.spinner("Analyzing the document..."):
                img_cv = np.array(image_to_process.convert('RGB'))
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                features = extract_lbp_features(img_cv)

                if features is not None:
                    features = features.reshape(1, -1)
                    prediction_idx = model.predict(features)
                    prediction_proba = model.predict_proba(features)
                    prediction_label = le.inverse_transform(prediction_idx)[0]
                    confidence = prediction_proba[0][prediction_idx[0]] * 100

                    st.success(f"**Prediction:** {prediction_label}")
                    st.info(f"**Confidence:** {confidence:.2f}%")

                    log_prediction(uploaded_file.name, prediction_label, confidence)
                    st.toast("Prediction logged!")

                    result_csv = f"Filename,Prediction,Confidence\n{uploaded_file.name},{prediction_label},{confidence:.2f}"
                    st.download_button(
                        label="Download Current Result (.csv)",
                        data=result_csv,
                        file_name=f"result_{uploaded_file.name}.csv",
                        mime='text/csv',
                    )
                    
                    st.write("### All Class Probabilities")
                    proba_dict = {le.classes_[i]: prediction_proba[0][i] * 100 for i in range(len(le.classes_))}
                    st.dataframe(
                        [proba_dict],
                        column_config={k: st.column_config.ProgressColumn(f"{k}", min_value=0, max_value=100, format="%.2f%%") for k in proba_dict},
                        hide_index=True
                    )
else:
    with col2:
        st.info("Please upload a document to begin the analysis.")