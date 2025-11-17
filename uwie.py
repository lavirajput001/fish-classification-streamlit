# uwie.py
import cv2
import numpy as np

def white_balance_simple(img):
    # Gray world assumption white balance
    result = img.copy().astype(np.float32)
    b,g,r = cv2.split(result)
    avgB = np.mean(b)
    avgG = np.mean(g)
    avgR = np.mean(r)
    avg = (avgB+avgG+avgR)/3.0
    b = b * (avg/avgB)
    g = g * (avg/avgG)
    r = r * (avg/avgR)
    result = cv2.merge([b,g,r])
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result

def UWIE(img):
    # img: BGR uint8
    wb = white_balance_simple(img)
    clahe = apply_clahe(wb)
    # optional smoothing: bilateral
    out = cv2.bilateralFilter(clahe, d=5, sigmaColor=75, sigmaSpace=75)
    return out
