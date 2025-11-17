# app.py
import streamlit as st
import cv2
import numpy as np
from infer import predict_image_streamlit
from PIL import Image

st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("Fish Species Classifier (UWIE + SURF/ORB + PatternNet)")

st.sidebar.header("Settings")
approach = st.sidebar.selectbox("Approach", ["cnn", "bovw"])
st.sidebar.write("Upload image to classify fish species.")

uploaded = st.file_uploader("Upload an underwater fish image", type=['jpg','jpeg','png'])
if uploaded is not None:
    # read image via PIL then convert to OpenCV BGR
    pil = Image.open(uploaded).convert('RGB')
    img = np.array(pil)[:, :, ::-1].copy()  # RGB->BGR
    st.image(pil, caption="Uploaded image", use_column_width=True)

    st.write("Running preprocessing and prediction...")
    with st.spinner("Predicting..."):
        label, conf = predict_image_streamlit(img, approach=approach)
    st.success(f"Predicted: {label}")
    if conf is not None:
        st.write(f"Confidence: {conf:.3f}")
    st.write("If confidence is low try another image or retrain model with more data.")
