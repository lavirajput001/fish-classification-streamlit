import numpy as np
import cv2

def final_predict(model, image, label_encoder):
    # If image is None -> error
    if image is None:
        raise ValueError("Image not loaded! Received None.")

    # Resize for CNN
    img = cv2.resize(image, (96, 96))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # (1,96,96,3)

    preds = model.predict(img)
    index = np.argmax(preds)
    label = label_encoder.inverse_transform([index])[0]
    return label
