import numpy as np

def final_predict(model, optimized_features, label_encoder):
    """
    Convert optimized features → reshape → model prediction → label decode
    """
    optimized_features = np.array(optimized_features, dtype=np.float32)

    # Reshape as 1 sample with feature vector
    optimized_features = optimized_features.reshape(1, -1)

    # Run prediction
    pred = model.predict(optimized_features)
    class_index = np.argmax(pred)

    # Decode label
    label = label_encoder.inverse_transform([class_index])[0]

    return label
