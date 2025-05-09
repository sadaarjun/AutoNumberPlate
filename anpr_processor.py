import cv2
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

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

        # Make a copy of the image to avoid modifying the original
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

        # Apply gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(blur)

        return equalized

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        # Return original image if preprocessing fails
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
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)

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

                # Check the aspect ratio of the rectangle (typical license plate aspect ratio)
                aspect_ratio = float(w) / h
                if 1.5 <= aspect_ratio <= 5.0:  # Typical license plate aspect ratios
                    # Extract the plate region
                    plate_img = image[y:y+h, x:x+w]
                    return plate_img

        return None

    except Exception as e:
        logger.error(f"Error detecting plate region: {str(e)}")
        return None

def recognize_plate_text(plate_img):
    """
    Recognize text on the license plate

    Args:
        plate_img: Cropped image of the license plate

    Returns:
        Recognized text or empty string if recognition failed
    """
    try:
        # This is a placeholder for actual OCR implementation
        # In a real application, you would integrate with a proper OCR library
        # such as Tesseract OCR or a specialized ANPR OCR engine

        # For this example, we'll return a mock plate number
        # In a real implementation, this would be replaced with actual OCR logic
        return "ABC123"

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
