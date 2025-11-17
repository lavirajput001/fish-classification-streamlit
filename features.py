# features.py
import cv2
import numpy as np

# try to create SURF; if not available fall back to ORB
def make_detector(surf_hessian=400):
    try:
        # xfeatures2d.SURF_create is present in opencv-contrib-python
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=surf_hessian)
        return ('surf', surf)
    except Exception as e:
        # fallback to ORB
        orb = cv2.ORB_create(nfeatures=1000)
        return ('orb', orb)

def extract_descriptors(img, detector_tuple):
    # img: BGR image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    name, detector = detector_tuple
    kp, des = detector.detectAndCompute(gray, None)
    if des is None:
        # empty descriptor: return zeros
        return np.zeros((1,128), dtype=np.float32)
    # normalize descriptors
    des = des.astype(np.float32)
    return des
