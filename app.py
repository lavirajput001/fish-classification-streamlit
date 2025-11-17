import streamlit as st
import cv2
import numpy as np
from infer import predict_image_streamlit
from PIL import Image

st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("🐟 Fish Classification App")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    approach = st.selectbox("Select approach", ["cnn", "bovw"])
    if st.button("Predict"):
        prediction = predict_image_streamlit(img_bgr, approach)
        st.success(f"Predicted Fish Species: {prediction}")
