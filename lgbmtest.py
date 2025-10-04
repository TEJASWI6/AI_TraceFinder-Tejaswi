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

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="TraceFinder")

# --- CONFIGURATION ---
MODEL_PATH = 'Baseline_model_v21.joblib'
ENCODER_PATH = 'Baseline_label_encoder_v21.joblib'
LOG_FILE = 'prediction_log.csv'
IMG_SIZE = 256

# --- Feature Extraction Function (Unchanged) ---
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

# --- Load Model and Encoder (Unchanged) ---
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

# --- LOGIC CHANGE: Removed 'auth_status' from logging ---
def log_prediction(filename, prediction, confidence, best_guess="N/A"):
    new_log_data = {
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Filename': [filename],
        'Prediction': [prediction],
        'Confidence': [f"{confidence:.2f}%"],
        'Best_Guess_Source': [best_guess]
    }
    log_columns = ['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Best_Guess_Source']
    new_log = pd.DataFrame(new_log_data, columns=log_columns)

    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        # Ensure all columns exist in the old log file, if not, add them
        for col in log_columns:
            if col not in log_df.columns:
                log_df[col] = "N/A"
        log_df = pd.concat([log_df, new_log], ignore_index=True)
    else:
        log_df = new_log
    
    log_df.to_csv(LOG_FILE, index=False)

# --- UI IMPROVEMENT: New App Interface ---
st.title("TraceFinder: Forensic Document Analysis 🕵️‍♂️")
st.markdown("Upload a document to identify its class (`Original`, `Tampered`, or the source scanner).")
st.divider()

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file...", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"], label_visibility="collapsed")

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
        img_cv = np.array(image_to_process.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        features = extract_lbp_features(img_cv)

        if features is not None:
            features = features.reshape(1, -1)
            prediction_idx = model.predict(features)
            prediction_proba = model.predict_proba(features)
            prediction_label = le.inverse_transform(prediction_idx)[0]
            confidence = prediction_proba[0][prediction_idx[0]] * 100
            
            # --- LOGIC CHANGE: This block is now only for calculating the best guess ---
            best_guess_label = "N/A"
            if prediction_label in ['Tampered', 'Original']:
                guess_probabilities = prediction_proba[0].copy()
                classes_to_ignore = ['Tampered', 'Original']
                for class_name in classes_to_ignore:
                    if class_name in le.classes_:
                        idx = np.where(le.classes_ == class_name)[0][0]
                        guess_probabilities[idx] = -1
                
                best_guess_index = np.argmax(guess_probabilities)
                best_guess_label = le.classes_[best_guess_index]
            
            # --- LOGIC CHANGE: Removed 'auth_status' from the log call ---
            log_prediction(uploaded_file.name, prediction_label, confidence, best_guess_label)
            st.toast("Prediction logged!")

            # --- UI IMPROVEMENT: New, cleaner display ---
            with col1:
                st.image(image_to_process, caption="Analyzed Image", use_column_width=True)
            
            with col2:
                st.header("Analysis Result")
                
                with st.container(border=True):
                    st.metric(label="Predicted Class", value=prediction_label)
                    st.caption(f"Confidence: {confidence:.2f}%")

                    # This is the highlighted best guess section
                    if prediction_label in ['Tampered', 'Original']:
                        st.divider()
                        st.metric(label="Source Scanner (Best Guess) 🔎", value=best_guess_label)
                
                st.write("") # Add some space
                
                st.subheader("All Class Probabilities")
                proba_dict = {le.classes_[i]: prediction_proba[0][i] * 100 for i in range(len(le.classes_))}
                st.dataframe(
                    [proba_dict],
                    column_config={k: st.column_config.ProgressColumn(f"{k}", min_value=0, max_value=100, format="%.2f%%") for k in proba_dict},
                    hide_index=True
                )
                
                st.write("") # Add some space

                result_csv = f"Filename,Prediction,Confidence,Best_Guess\n{uploaded_file.name},{prediction_label},{confidence:.2f},{best_guess_label}"
                st.download_button(
                    label="Download Current Result (.csv)",
                    data=result_csv,
                    file_name=f"result_{uploaded_file.name}.csv",
                    mime='text/csv',
                )

# Sidebar (UI Tweaks for cleanliness)
st.sidebar.title("📜 Prediction History")
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.sidebar.dataframe(log_df, use_container_width=True)
    st.sidebar.download_button(
        label="Download Full Log (.csv)",
        data=log_df.to_csv(index=False).encode('utf-8'),
        file_name="full_prediction_log.csv",
        mime='text/csv'
    )
else:
    st.sidebar.info("No predictions have been logged yet.")

# Placeholder message
if uploaded_file is None:
    with col2:
        st.info("Please upload a document to begin the analysis.")

