import cv2
import numpy as np
import logging
import re
import pytesseract
import os

logger = logging.getLogger(__name__)

class ANPRSettings:
    """Class to hold ANPR configuration settings"""
    def __init__(self):
        self.enable_preprocessing = True
        self.min_plate_size = 200  # Lowered for smaller plates
        self.max_plate_size = 20000  # Lowered for resized images

def process_anpr(image, anpr_settings):
    """
    Process an image for license plate detection and recognition
    """
    try:
        if image is None or image.size == 0:
            return False, "Invalid image input"

        max_width = 800
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / width
            image = cv2.resize(image, (max_width, int(height * ratio)))

        img = image.copy()

        if anpr_settings.enable_preprocessing:
            img = preprocess_image(img)

        plate_img = detect_plate_region(img, anpr_settings.min_plate_size, anpr_settings.max_plate_size)

        if plate_img is None:
            return False, "No license plate detected in the image"

        plate_text = recognize_plate_text(plate_img)

        if not plate_text:
            return False, "Could not recognize text on the license plate"

        cleaned_plate = clean_plate_text(plate_text)

        if not cleaned_plate:
            return False, "Recognized text does not appear to be a valid license plate"

        return True, cleaned_plate

    except Exception as e:
        logger.error(f"Error in ANPR processing: {str(e)}")
        return False, f"ANPR Processing Error: {str(e)}"

def preprocess_image(image):
    """
    Apply preprocessing to enhance the image for license plate detection
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Increased kernel for smoother edges
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)  # Inverted for better edges
        # Save preprocessed image for debugging
        cv2.imwrite("debug/preprocessed.jpg", thresh)
        return thresh

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def detect_plate_region(image, min_plate_size, max_plate_size, debug_dir="debug"):
    """
    Detect license plate region in the image
    """
    try:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        median_intensity = np.median(image)
        lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
        upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

        edges = cv2.Canny(image, lower_threshold, upper_threshold)
        cv2.imwrite(os.path.join(debug_dir, "edges.jpg"), edges)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        plate_img = None

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_plate_size or area > max_plate_size:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

            if 3 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 1.0 <= aspect_ratio <= 6.0:
                    plate_img = image[y:y+h, x:x+w]
                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"Area: {area:.0f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    break

        cv2.imwrite(os.path.join(debug_dir, "contours.jpg"), debug_img)
        if plate_img is not None:
            cv2.imwrite(os.path.join(debug_dir, "detected_plate.jpg"), plate_img)

        return plate_img

    except Exception as e:
        logger.error(f"Error detecting plate region: {str(e)}")
        return None

def recognize_plate_text(plate_img):
    """
    Recognize text on the license plate using Tesseract OCR
    """
    try:
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(plate_img, config=custom_config)
        text = text.strip()
        return text if text else ""

    except Exception as e:
        logger.error(f"Error in plate text recognition: {str(e)}")
        return ""

def clean_plate_text(plate_text):
    """
    Clean and validate the recognized plate text
    """
    try:
        cleaned = plate_text.replace(" ", "").upper()
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        if len(cleaned) >= 5:
            return cleaned
        else:
            return ""

    except Exception as e:
        logger.error(f"Error cleaning plate text: {str(e)}")
        return ""