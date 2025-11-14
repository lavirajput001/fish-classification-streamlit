import cv2
import numpy as np

def extract_orb_features(img):
    """
    ORB extract → descriptor flatten → return feature vector
    """
    orb = cv2.ORB_create(nfeatures=500)

    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None:
        descriptors = np.zeros((1, 32))

    # Flatten feature vector
    features = descriptors.flatten()

    # Limit size for fixed input dimension
    max_len = 1024

    if len(features) < max_len:
        features = np.pad(features, (0, max_len - len(features)), mode='constant')
    else:
        features = features[:max_len]

    return np.array(features)
