import cv2
import numpy as np

def enhance_image(img):
    """
    UWIE: White balance + CLAHE enhancement
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0)
    l_clahe = clahe.apply(l)

    # Merge and convert back
    enhanced = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced
