import cv2
import numpy as np

def make_detector():
    return cv2.ORB_create(nfeatures=500)

def extract_descriptors(detector, img_gray):
    keypoints, descriptors = detector.detectAndCompute(img_gray, None)
    return keypoints, descriptors

def compute_bovw_h(descriptors, vocab=None):
    if descriptors is None:
        return np.zeros((1, 500))
    return np.random.rand(1, 500)
