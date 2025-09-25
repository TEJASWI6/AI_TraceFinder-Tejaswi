import os
from datetime import datetime
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pdf2image import convert_from_bytes
from PIL import Image
from skimage.feature import local_binary_pattern

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- CONFIGURATION ---
SOURCE_MODEL_PATH = 'source_model_final_new_one.joblib'
SOURCE_ENCODER_PATH = 'source_model_encoder_final_new_two.joblib'
INTEGRITY_MODEL_PATH = 'integrity_model_cnn_new_three.h5'
LOG_FILE = 'prediction_log.csv'
IMG_SIZE_LBP = 256
IMG_SIZE_CNN = 128

# --- Session State Initialization ---
if 'last_processed_file' not in st.session_state:
    st.session_state.last_processed_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'toast_shown' not in st.session_state:
    st.session_state.toast_shown = True


# --- Feature Extraction ---
def extract_lbp_features(image_cv):
    try:
        img_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_SIZE_LBP, IMG_SIZE_LBP))
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_resized, n_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    except Exception as e:
        st.error(f"Error in LBP feature extraction: {e}")
        return None

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        source_model = joblib.load(SOURCE_MODEL_PATH)
        source_encoder = joblib.load(SOURCE_ENCODER_PATH)
        integrity_model = tf.keras.models.load_model(INTEGRITY_MODEL_PATH)
        return source_model, source_encoder, integrity_model
    except FileNotFoundError:
        st.error("One or more model/encoder files not found. Please check the filenames.")
        return None, None, None

source_model, source_le, integrity_model = load_models()

# --- Logging Function ---
def log_prediction(filename, status, source_guess, confidence):
    new_log = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Filename': [filename],
        'Status': [status],
        'Source_Guess': [source_guess],
        'Confidence': [f"{confidence:.2f}%"]
    })
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = pd.concat([log_df, new_log], ignore_index=True)
    else:
        log_df = new_log
    log_df.to_csv(LOG_FILE, index=False)

# --- App Interface ---
st.title("TraceFinder: Forensic Document Analysis 🕵️‍♂️")
st.write("Upload a document to check its authenticity and identify its source scanner.")

# --- Sidebar ---
st.sidebar.title("History")
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.sidebar.dataframe(log_df.iloc[::-1])
    st.sidebar.download_button("Download Full Log (.csv)", log_df.to_csv(index=False).encode('utf-8'), "full_log.csv", "text/csv")
else:
    st.sidebar.info("No predictions logged yet.")

# --- Main Page Layout ---
col1, col2 = st.columns(2)
with col1:
    st.header("Upload Document")
    def on_file_change():
        st.session_state.last_processed_file = None
        st.session_state.analysis_results = None
        st.session_state.toast_shown = False

    uploaded_file = st.file_uploader(
        "Choose a file...",
        type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"],
        on_change=on_file_change
    )

# --- Analysis Trigger ---
if uploaded_file is not None and st.session_state.analysis_results is None:
    pil_image = None
    if uploaded_file.type == "application/pdf":
        try:
            images = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=1)
            if images:
                pil_image = images[0].convert('RGB')
        except Exception as e:
            st.error(f"Error converting PDF: {e}")
    else:
        pil_image = Image.open(uploaded_file).convert('RGB')

    if pil_image is not None:
        with st.spinner("Analyzing..."):
            img_cv_for_lbp = np.array(pil_image)
            img_cv_for_lbp = cv2.cvtColor(img_cv_for_lbp, cv2.COLOR_RGB2BGR)

            img_for_cnn = pil_image.resize((IMG_SIZE_CNN, IMG_SIZE_CNN))
            img_array_cnn = tf.keras.preprocessing.image.img_to_array(img_for_cnn)
            img_array_cnn = np.expand_dims(img_array_cnn, axis=0) / 255.0
            integrity_pred_val = integrity_model.predict(img_array_cnn)[0][0]
            is_tampered = integrity_pred_val > 0.5
            integrity_status = "Tampered" if is_tampered else "Original"
            integrity_confidence = (integrity_pred_val if is_tampered else 1 - integrity_pred_val) * 100

            lbp_features = extract_lbp_features(img_cv_for_lbp)
            source_pred_label = "Unknown"
            source_probas = None
            if lbp_features is not None:
                lbp_features_reshaped = lbp_features.reshape(1, -1)
                source_pred_idx = source_model.predict(lbp_features_reshaped)
                source_pred_label = source_le.inverse_transform(source_pred_idx)[0]
                source_probas = source_model.predict_proba(lbp_features_reshaped)

            st.session_state.analysis_results = {
                'pil_image': pil_image,
                'integrity_status': integrity_status,
                'integrity_confidence': integrity_confidence,
                'source_pred_label': source_pred_label,
                'is_tampered': is_tampered,
                'source_probas': source_probas,
            }
            st.session_state.last_processed_file = uploaded_file.name

            log_prediction(uploaded_file.name, integrity_status, source_pred_label, integrity_confidence)
            st.rerun()


# --- Display Area ---
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results

    with col1:
        st.image(results['pil_image'], caption="Image being analyzed.", use_container_width=True)

    with col2:
        st.header("Analysis Result")
        if not st.session_state.toast_shown:
            st.toast("Analysis complete!")
            st.session_state.toast_shown = True

        if results['is_tampered']:
            st.error(f"**Status: {results['integrity_status']}**")
            st.info(f"**Confidence:** {results['integrity_confidence']:.2f}%")
            st.subheader("Source Analysis (Best Guess Based on Texture)")

            if results['source_probas'] is not None:
                source_probas = results['source_probas']
                proba_dict = {source_le.classes_[i]: source_probas[0][i] * 100 for i in range(len(source_le.classes_))}
                st.dataframe(
                    [proba_dict],
                    column_config={k: st.column_config.ProgressColumn(f"{k}", min_value=0, max_value=100, format="%.2f%%") for k in proba_dict},
                    hide_index=True
                )
                # --- NEW LINE ADDED HERE ---
                st.info(f"**Best Guess:** {results['source_pred_label']}")
        else:
            st.success(f"**Status: {results['integrity_status']}**")
            st.info(f"**Predicted Source:** {results['source_pred_label']}")
