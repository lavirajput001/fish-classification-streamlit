import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import pickle

# Import local modules
from uwie import enhance_image
from morph import morph_process
from features import extract_orb_features
from firefly import firefly_optimize
from infer import final_predict

# Load model
model = tf.keras.models.load_model("patternnet_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ---------------------------
# STREAMLIT UI STARTS HERE
# ---------------------------

st.title("🐟 Fish Species Classification (Research Paper Implementation)")
st.write("Upload an underwater fish image to classify using SURF/ORB + Firefly + PatternNet CNN")

uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.subheader("1️⃣ Underwater Image Enhancement (UWIE)")
    enhanced = enhance_image(img)
    st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="Enhanced Image", use_column_width=True)

    st.subheader("2️⃣ Morphological Processing")
    morph_img = morph_process(enhanced)
    st.image(morph_img, caption="Morph Processed Image", use_column_width=True)

    st.subheader("3️⃣ Feature Extraction (ORB/SURF Equivalent)")
    features = extract_orb_features(morph_img)
    st.write(f"Extracted Features Shape: {features.shape}")

    st.subheader("4️⃣ Firefly Optimization")
    optimized_features = firefly_optimize(features)
    st.write("Selected Optimized Features:", optimized_features.shape)

    st.subheader("5️⃣ PatternNet Prediction")
    predicted_label = final_predict(model, image, label_encoder)

    st.success(f"🎯 Predicted Fish Species: **{predicted_label}**")
