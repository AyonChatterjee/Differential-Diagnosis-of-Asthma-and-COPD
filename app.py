import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time

# Load trained model
model = load_model("audio_classifier.h5")

# Function to extract MFCC features from uploaded audio
def extract_features(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        feature = np.mean(mfcc.T, axis=0)
        return feature
    except Exception as e:
        st.error(f"⚠️ Error in feature extraction: {e}")
        return None

# Streamlit UI Customization
st.set_page_config(page_title="Audio Disease Classifier", page_icon="🎤", layout="centered")
st.title("🎤 Audio Disease Classification (Asthma vs. COPD)")
st.markdown("### Upload an audio file to predict the condition")

# Sidebar customization
st.sidebar.header("ℹ️ Instructions")
st.sidebar.info("1. Upload a WAV audio file.\n2. The model will analyze and classify the condition.\n3. You will receive a prediction with confidence score.")

# File uploader
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"], help="Only .wav files are supported")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.markdown("⌛ **Processing audio...**")
    
    with st.spinner("Extracting features..."):
        time.sleep(2)
        features = extract_features(uploaded_file)
    
    if features is not None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        
        with st.spinner("Analyzing with AI model..."):
            time.sleep(2)
            prediction = model.predict(features_scaled)[0][0]
            confidence = round(float(prediction) * 100, 2)
            result = "Asthma" if prediction > 0.5 else "COPD"
            
        st.success(f"✅ **Predicted Condition: {result}**")
        st.progress(int(confidence))
        st.info(f"🧠 Confidence Score: **{confidence}%**")
