# frontend.py
import streamlit as st
import torch
import librosa
import numpy as np
import os
from app import CNNModel, extract_features, load_model, device

# ================================
# Load Trained Model
# ================================
MODEL_PATH = "saved_model.pth"
model = load_model(MODEL_PATH)

# ================================
# Streamlit UI Setup
# ================================
st.set_page_config(
    page_title="DeepFake Voice Detector",
    page_icon="🎧",
    layout="centered",
)

# ================================
# Custom CSS for Clean Look
# ================================
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: white;
            font-family: 'Trebuchet MS';
        }
        h1 {
            color: #00ADB5;
        }
        .upload-box {
            border: 2px dashed #00ADB5;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# Page Header
# ================================
st.title("🎧 DeepFake Voice Detection System")
st.subheader("Detect whether an audio is 🧑 Human or 🤖 AI-generated")
st.markdown("---")

# ================================
# File Upload
# ================================
uploaded_file = st.file_uploader("Upload an audio file (.flac or .wav)", type=["flac", "wav"])

if uploaded_file is not None:
    file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/wav")

    # ================================
    # Run Prediction
    # ================================
    st.markdown("---")
    st.write("🔍 **Analyzing Audio...**")

    try:
        features = extract_features(file_path)
        features = torch.tensor(features, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(features)
            pred = torch.argmax(output, dim=1).item()

        result = "🧑 Bonafide Human Voice" if pred == 0 else "🤖 AI/Deepfake Voice"

        st.markdown(f"### ✅ Prediction: {result}")
        st.markdown("---")

    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")

else:
    st.info("⬆️ Please upload a .flac or .wav audio file to start detection.")

# ================================
# Footer
# ================================
st.markdown(
    """
    <br><hr>
    <center>
    Made with ❤️ using <b>Streamlit</b> & <b>PyTorch</b><br>
    <small>Project: AI vs Human Voice Detection</small>
    </center>
    """,
    unsafe_allow_html=True,
)
