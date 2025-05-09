import cv2
import numpy as np
import logging
import re
import pytesseract

logger = logging.getLogger(__name__)

class ANPRSettings:
    """Class to hold ANPR configuration settings"""
    def __init__(self):
        self.enable_preprocessing = True
        self.min_plate_size = 300  # Adjusted for smaller images
        self.max_plate_size = 30000  # Adjusted for smaller images

def process_anpr(image, anpr_settings):
    """
    Process an image for license plate detection and recognition

    Args:
        image: The captured image as a numpy array
        anpr_settings: ANPRSettings object with ANPR configuration

    Returns:
        Tuple (success, result) where:
            - success is a boolean indicating if a plate was successfully detected
            - result is either the plate text (on success) or an error message (on failure)
    """
    try:
        if image is None or image.size == 0:
            return False, "Invalid image input"

        # Resize image to reduce processing load (max width 800px)
        max_width = 800
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / width
            image = cv2.resize(image, (max_width, int(height * ratio)))

        # Make a copy of the image
        img = image.copy()

        # Apply preprocessing if enabled
        if anpr_settings.enable_preprocessing:
            img = preprocess_image(img)

        # Detect plate region
        plate_img = detect_plate_region(img, anpr_settings.min_plate_size, anpr_settings.max_plate_size)

        if plate_img is None:
            return False, "No license plate detected in the image"

        # Recognize the text
        plate_text = recognize_plate_text(plate_img)

        if not plate_text:
            return False, "Could not recognize text on the license plate"

        # Clean and validate the plate text
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

    Args:
        image: Input image

    Returns:
        Preprocessed image
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply light gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  # Smaller kernel for faster processing

        # Apply adaptive thresholding instead of histogram equalization for better contrast
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        return thresh

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

def detect_plate_region(image, min_plate_size, max_plate_size):
    """
    Detect license plate region in the image

    Args:
        image: Preprocessed image
        min_plate_size: Minimum plate contour area
        max_plate_size: Maximum plate contour area

    Returns:
        Cropped plate image or None if no plate found
    """
    try:
        # Apply edge detection with adjusted thresholds
        edges = cv2.Canny(image, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Check each contour
        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip if contour is too small or too large
            if area < min_plate_size or area > max_plate_size:
                continue

            # Get the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)

            # Approximate the contour shape
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # If the contour has 4 points (rectangular), it might be a license plate
            if len(approx) == 4:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check the aspect ratio of the rectangle
                aspect_ratio = float(w) / h
                if 1.5 <= aspect_ratio <= 5.0:
                    # Extract the plate region
                    plate_img = image[y:y+h, x:x+w]
                    return plate_img

        return None

    except Exception as e:
        logger.error(f"Error detecting plate region: {str(e)}")
        return None

def recognize_plate_text(plate_img):
    """
    Recognize text on the license plate using Tesseract OCR

    Args:
        plate_img: Cropped image of the license plate

    Returns:
        Recognized text or empty string if recognition failed
    """
    try:
        # Resize plate image for better OCR accuracy
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Configure Tesseract with custom settings for license plates
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(plate_img, config=custom_config)

        # Remove any newlines or extra spaces
        text = text.strip()

        return text if text else ""

    except Exception as e:
        logger.error(f"Error in plate text recognition: {str(e)}")
        return ""

def clean_plate_text(plate_text):
    """
    Clean and validate the recognized plate text

    Args:
        plate_text: Raw recognized text

    Returns:
        Cleaned plate text or empty string if validation failed
    """
    try:
        # Remove spaces and convert to uppercase
        cleaned = plate_text.replace(" ", "").upper()

        # Remove any special characters
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

        # Basic validation - must have at least 5 characters
        if len(cleaned) >= 5:
            return cleaned
        else:
            return ""

    except Exception as e:
        logger.error(f"Error cleaning plate text: {str(e)}")
        return ""