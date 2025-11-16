import numpy as np
import cv2

def final_predict(model, image, label_encoder):

    # Resize image for CNN
    img = cv2.resize(image, (96, 96))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # (1,96,96,3)

    preds = model.predict(img)
    index = np.argmax(preds)

    # Convert index → label
    label = label_encoder.inverse_transform([index])[0]
    return label
