import cv2

def morph_process(img):
    """
    Morphological noise removal + sharpening
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    sharp = cv2.GaussianBlur(opening, (0, 0), 3)
    sharp = cv2.addWeighted(opening, 1.5, sharp, -0.5, 0)

    return sharp
