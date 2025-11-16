import cv2
import numpy as np

def extract_orb_features(image):
    """
    Extract ORB keypoints + descriptors
    Convert descriptors into 1D fixed-size vector
    """

    # ORB extractor (500 features)
    orb = cv2.ORB_create(nfeatures=500)

    # Detect keypoints + descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # If no descriptors found → return zero vector
    if descriptors is None:
        return np.zeros(500 * 32)   # ORB descriptor = 32 bytes

    # Flatten descriptors
    desc = descriptors.flatten()

    # Make fixed-length vector (500 descriptors × 32 = 16000)
    max_length = 500 * 32

    if desc.shape[0] < max_length:
        # Pad with zeros
        padded = np.zeros(max_length)
        padded[:desc.shape[0]] = desc
        desc = padded
    else:
        # Trim extra values
        desc = desc[:max_length]

    # Final shape → (16000,)
    return desc.astype(np.float32)
