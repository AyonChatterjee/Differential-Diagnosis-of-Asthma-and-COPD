import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

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
        st.error(f"Error in feature extraction: {e}")
        return None

# Streamlit UI
st.title("🎤 Audio Disease Classification (Asthma vs. COPD)")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Extract features
    features = extract_features(uploaded_file)
    
    if features is not None:
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])  # Reshape for model
        
        # Reshape for CNN input
        features_scaled = np.expand_dims(features_scaled, axis=-1)
        
        # Prediction
        prediction = model.predict(features_scaled)[0][0]
        result = "Asthma" if prediction > 0.5 else "COPD"
        
        st.success(f"🔍 Predicted Condition: **{result}**")

