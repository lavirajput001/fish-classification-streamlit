import numpy as np
import cv2

def final_predict(model, image, label_encoder):
    img = cv2.resize(image, (96, 96))   # CNN input size
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)   # (1, 96, 96, 3)

    pred = model.predict(img)
    index = np.argmax(pred)
    label = label_encoder.inverse_transform([index])[0]
    return label
