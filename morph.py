# morph.py
import cv2
import numpy as np

def morph_process(img):
    # img: BGR uint8 -> convert to gray and apply morphological ops
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to segment foreground (fish)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 7)
    # clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    # optional contour refine: keep largest contour as fish
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
        # smooth mask
        mask = cv2.medianBlur(mask, 7)
    # return masked image (BGR) and mask
    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask
