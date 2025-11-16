import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle

from infer import final_predict   # we use the simplified correct infer.py

# ------------------------------
# Load Model + Label Encoder
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("patternnet_model.h5")
    return model

@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return le

model = load_model()
label_encoder = load_label_encoder()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🐟 Fish Species Classification (PatternNet CNN)")
st.write("Upload a fish image and get prediction.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# ------------------------------
# Handle Image Upload
# ------------------------------
if uploaded_file is not None:

    # Convert uploaded file → OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Could not read the image. Upload a valid file.")
    else:
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # --------------------------
        # Predict using CNN
        # --------------------------
        try:
            predicted_label = final_predict(model, image, label_encoder)
            st.success(f"🎉 **Predicted Species:** {predicted_label}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
