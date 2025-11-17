import cv2
import numpy as np
import pickle
from uwie import UWIE
from morph import morph_process
from features import make_detector, extract_descriptors, compute_bovw_h
from tensorflow.keras.models import load_model

# Load Keras model
model = load_model("patternnet_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize feature detector
detector = make_detector()

def predict_image_streamlit(img_bgr, approach='cnn'):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if approach == 'bovw':
        _, descriptors = extract_descriptors(detector, img_gray)
        features = compute_bovw_h(descriptors)
        # dummy prediction for BoVW
        pred_class = "Fish_BoVW"
    else:  # CNN
        img_resized = cv2.resize(img_bgr, (224, 224))
        img_input = np.expand_dims(img_resized / 255.0, axis=0)
        pred_probs = model.predict(img_input)
        pred_class = le.inverse_transform([np.argmax(pred_probs)])[0]

    return pred_class
