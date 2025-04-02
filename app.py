import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import requests

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

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
loading_animation = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_j1adxtyb.json")
result_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_puciaact.json")

# Streamlit UI Customization
st.set_page_config(page_title="Audio Disease Classifier", page_icon="🫁", layout="wide")

st.markdown("""
    <style>
        .title-container {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #4CAF50;
        }
        .upload-box {
            text-align: center;
            padding: 20px;
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

# Banner with Lung Image
st.image("https://www.cdc.gov/copd/images/COPD_Banner_1200x675.jpg", use_column_width=True)

st.markdown("<div class='title-container'>🫁 Lung Disease Detection (Asthma vs. COPD)</div>", unsafe_allow_html=True)
st.markdown("### Upload an audio file to predict the condition")

# Sidebar customization
st.sidebar.header("ℹ️ Instructions")
st.sidebar.info("1. Upload a WAV audio file.\n2. The model will analyze and classify the condition.\n3. You will receive a prediction with confidence score.")

# File uploader
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"], help="Only .wav files are supported")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.markdown("⌛ **Processing audio...**")
    st_lottie(loading_animation, height=150, key="loading")
    
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
        st_lottie(result_animation, height=200, key="result")
        st.progress(int(confidence))
        st.info(f"🧠 Confidence Score: **{confidence}%**")
        
        # Display relevant lung image based on prediction
        if result == "Asthma":
            st.image("https://www.verywellhealth.com/thmb/6EuKwzFTVjzFlm-XR11OdIBKrXc=/3000x2000/filters:fill(87E3EF,1)/lungs-with-asthma-56a79cbf3df78cf7729c77e8.jpg", width=500, caption="Lungs affected by Asthma")
        else:
            st.image("https://www.verywellhealth.com/thmb/l_YPWQzOGVozAykW-QCO97qW4Hc=/3000x2000/filters:fill(87E3EF,1)/copd-lungs-56a79cbf3df78cf7729c77e7.jpg", width=500, caption="Lungs affected by COPD")
