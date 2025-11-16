import streamlit as st
from features import extract_features
from model import load_model, predict
import pickle
import numpy as np

# Load model
model = load_model("patternnet_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Fish Classification App")

# Image upload
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Feature extraction
    features = extract_features(uploaded_file)
    
    # Prediction
    prediction = predict(model, features)
    
    # ✅ This is where you add the fix
    prediction = np.array(prediction).reshape(-1)  # flatten to 1D
    predicted_label = le.inverse_transform(prediction)[0]
    
    st.success(f"Predicted Species: {predicted_label}")
