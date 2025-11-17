import streamlit as st
import pickle
from PIL import Image
import numpy as np

# ----------------------------
# Import your actual functions
# ----------------------------
from features import extract_features
from infer import predict_image_streamlit  # ensure infer.py has this function
from model import load_model  # ensure model.py has load_model function

st.set_page_config(page_title="Fish Classification App", page_icon="🐟")
st.title("Fish Classification App")
st.write("Upload an image of a fish to classify its species.")

# ----------------------------
# Load trained model
# ----------------------------
model = load_model("patternnet_model.h5")

# ----------------------------
# Load label encoder
# ----------------------------
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ----------------------------
# Image upload section
# ----------------------------
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg","png"])
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract features using your features.py
    features = extract_features(image)
    
    # Predict using your infer.py
    prediction = predict_image_streamlit(model, features)
    
    # Ensure prediction is 1D array
    prediction = np.array(prediction).reshape(-1)
    
    # Decode label using label_encoder.pkl
    predicted_label = le.inverse_transform(prediction)[0]
    
    st.success(f"Predicted Species: {predicted_label}")
