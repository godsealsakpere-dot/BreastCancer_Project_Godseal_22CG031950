import streamlit as st
import joblib
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Custom CSS for styling
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Point to the static css file
local_css("static/style.css")

# Title and Overview
st.title("🎗️ Breast Cancer Prediction System")
st.write("""
This system predicts whether a breast mass is **Benign** or **Malignant** based on FNA analysis.
*Note: This tool is for educational purposes only.*
""")

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'breast_cancer_model.pkl')

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please run Part A to generate the model.")
    st.stop()

# Create Input Form
st.subheader("Enter Tumor Features")

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0, format="%.2f")
    texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0, format="%.2f")
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=90.0, format="%.2f")

with col2:
    area_mean = st.number_input("Area Mean", min_value=0.0, value=650.0, format="%.2f")
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1, format="%.4f")

# Prediction Button
if st.button("Analyze Tumor"):
    # Prepare input array (must match training order)
    # Features: ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    
    # Map prediction (0 = Malignant, 1 = Benign in sklearn dataset)
    # logic: if 0 -> Malignant, if 1 -> Benign
    if prediction == 0:
        result = "Malignant"
        st.error(f"The prediction is: **{result}**")
        st.write("⚠️ This indicates a cancerous tumor.")
    else:
        result = "Benign"
        st.success(f"The prediction is: **{result}**")
        st.write("✅ This indicates a non-cancerous tumor.")