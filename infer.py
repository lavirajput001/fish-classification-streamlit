import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# -----------------------------
# LOAD MODEL + LABELS
# -----------------------------
model = load_model("patternnet_model.h5")  # ya fish_model.h5
img_size = (224, 224)   # <- IMPORTANT : match your model size

with open("label_encoder.pkl", "rb") as f:
    class_indices = pickle.load(f)

labels = {v: k for k, v in class_indices.items()}


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image_streamlit(img_bgr, approach="cnn"):
    try:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)        # <- FIXED
        x = img.astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        proba = model.predict(x)[0]           # predicts correctly
        idx = np.argmax(proba)
        label = labels[idx]
        confidence = float(proba[idx])

        return label, confidence

    except Exception as e:
        return f"Error: {str(e)}", 0.0
