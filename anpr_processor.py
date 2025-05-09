import cv2
import numpy as np
import logging
import re
import pytesseract
import os

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anpr_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ANPRSettings:
    """Class to hold ANPR configuration settings"""
    def __init__(self):
        self.enable_preprocessing = True
        self.min_plate_size = 10  # Lowered for small/slanted plates
        self.max_plate_size = 100000  # Increased for flexibility

def process_anpr(image, anpr_settings):
    """
    Process an image for HSRP license plate detection and recognition
    """
    try:
        logger.debug("Starting ANPR processing")
        if image is None or image.size == 0:
            logger.error("Invalid image input: Image is None or empty")
            return False, "Invalid image input"

        height, width = image.shape[:2]
        logger.debug(f"Input image dimensions: {width}x{height}")

        max_width = 800
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            image = cv2.resize(image, (max_width, new_height))
            logger.debug(f"Resized image to: {max_width}x{new_height}")
        else:
            logger.debug("No resizing needed")

        img = image.copy()

        if anpr_settings.enable_preprocessing:
            logger.debug("Applying preprocessing")
            img = preprocess_image(img)
        else:
            logger.debug("Preprocessing disabled")

        logger.debug("Detecting plate region")
        plate_img = detect_plate_region(img, anpr_settings.min_plate_size, anpr_settings.max_plate_size)

        if plate_img is None:
            logger.warning("No license plate detected in the image")
            return False, "No license plate detected in the image"

        logger.debug("Recognizing plate text")
        plate_text = recognize_plate_text(plate_img)

        if not plate_text:
            logger.warning("Could not recognize text on the license plate")
            return False, "Could not recognize text on the license plate"

        logger.debug(f"Raw plate text: {plate_text}")
        cleaned_plate = clean_plate_text(plate_text)

        if not cleaned_plate:
            logger.warning("Recognized text does not appear to be a valid HSRP license plate")
            return False, "Recognized text does not appear to be a valid HSRP license plate"

        logger.info(f"Successfully detected license plate: {cleaned_plate}")
        return True, cleaned_plate

    except Exception as e:
        logger.error(f"Error in ANPR processing: {str(e)}", exc_info=True)
        return False, f"ANPR Processing Error: {str(e)}"

def preprocess_image(image):
    """
    Apply preprocessing to enhance the image for HSRP license plate detection
    """
    try:
        logger.debug("Starting image preprocessing")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        logger.debug("Converted to HSV for color filtering")

        # Minimal color filtering for HSRP plates (white, yellow, green)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_yellow, mask_green))
        logger.debug("Applied color filtering for white, yellow, green backgrounds")

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted to grayscale after color filtering")

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        logger.debug("Applied Gaussian blur with kernel (5,5)")

        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        logger.debug("Applied adaptive thresholding (inverted, blockSize=11)")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)  # Light dilation
        logger.debug("Applied dilation to strengthen plate edges")

        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "preprocessed.jpg"), thresh)
        cv2.imwrite(os.path.join(debug_dir, "color_mask.jpg"), mask)
        logger.debug("Saved preprocessed and color mask images to debug/")

        return thresh

    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}", exc_info=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        logger.debug("Falling back to grayscale image due to preprocessing error")
        return gray

def correct_perspective(image, contour):
    """
    Apply perspective correction to a quadrilateral contour to obtain a rectangular plate
    """
    try:
        logger.debug("Applying perspective correction")
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)  # Increased epsilon

        if len(approx) < 4:
            logger.debug(f"Not enough points for perspective correction: {len(approx)}")
            x, y, w, h = cv2.boundingRect(contour)
            return image[y:y+h, x:x+w]

        pts = approx.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        if rect[0][1] > rect[3][1] or rect[1][1] > rect[2][1]:
            logger.debug("Invalid point ordering, adjusting")
            temp = rect[0].copy()
            rect[0] = rect[3]
            rect[3] = rect[1]
            rect[1] = rect[2]
            rect[2] = temp

        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        logger.debug(f"Perspective corrected to {max_width}x{max_height}")

        cv2.imwrite(os.path.join("debug", "warped_plate.jpg"), warped)
        logger.debug("Saved warped plate to debug/warped_plate.jpg")

        return warped

    except Exception as e:
        logger.error(f"Error in perspective correction: {str(e)}", exc_info=True)
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]

def detect_plate_region(image, min_plate_size, max_plate_size, debug_dir="debug"):
    """
    Detect HSRP license plate region in the image
    """
    try:
        logger.debug("Starting plate region detection")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            logger.debug(f"Created debug directory: {debug_dir}")

        height, width = image.shape[:2]
        logger.debug(f"Input image for detection: {width}x{height}")

        median_intensity = np.median(image)
        lower_threshold = int(max(0, (1.0 - 0.4) * median_intensity))
        upper_threshold = int(min(255, (1.0 + 0.4) * median_intensity))
        logger.debug(f"Canny thresholds: lower={lower_threshold}, upper={upper_threshold}")

        edges = cv2.Canny(image, lower_threshold, upper_threshold)
        cv2.imwrite(os.path.join(debug_dir, "edges.jpg"), edges)
        logger.debug("Saved edge-detected image to debug/edges.jpg")

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"Found {len(contours)} contours")

        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]  # Increased
        logger.debug(f"Considering top {len(contours)} contours (max 40)")

        plate_img = None
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_plate_size:
                logger.debug(f"Contour {i}: Skipped (area={area:.0f} < min={min_plate_size})")
                continue
            if area > max_plate_size:
                logger.debug(f"Contour {i}: Skipped (area={area:.0f} > max={max_plate_size})")
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            num_points = len(approx)

            if num_points < 3 or num_points > 12:  # More flexible
                logger.debug(f"Contour {i}: Skipped (points={num_points}, expected 3-12)")
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 10.0:  # Wider range for HSRP
                logger.debug(f"Contour {i}: Skipped (aspect_ratio={aspect_ratio:.2f}, expected 0.5-10.0)")
                continue

            logger.debug(
                f"Contour {i}: Valid (area={area:.0f}, points={num_points}, aspect_ratio={aspect_ratio:.2f})"
            )

            plate_img = correct_perspective(image, contour)
            cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 2)
            cv2.putText(
                debug_img, f"Area: {area:.0f}, AR: {aspect_ratio:.2f}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            break

        cv2.imwrite(os.path.join(debug_dir, "contours.jpg"), debug_img)
        logger.debug("Saved contour image to debug/contours.jpg")

        if plate_img is None:
            logger.warning("No valid plate region found after filtering")
        else:
            cv2.imwrite(os.path.join(debug_dir, "detected_plate.jpg"), plate_img)
            logger.debug("Saved detected plate to debug/detected_plate.jpg")

        return plate_img

    except Exception as e:
        logger.error(f"Error detecting plate region: {str(e)}", exc_info=True)
        return None

def recognize_plate_text(plate_img):
    """
    Recognize text on HSRP license plate using Tesseract OCR
    """
    try:
        logger.debug("Starting plate text recognition")
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)  # Enhance contrast
        logger.debug("Applied CLAHE contrast enhancement")

        blur = cv2.bilateralFilter(enhanced, 11, 17, 17)
        logger.debug("Applied bilateral filter for OCR")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        logger.debug("Applied morphological closing")

        binary = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6
        )
        logger.debug("Applied adaptive thresholding for OCR (blockSize=25)")

        binary = cv2.fastNlMeansDenoising(binary, h=15)
        logger.debug("Applied denoising")

        binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        logger.debug("Resized plate image for OCR (4x)")

        cv2.imwrite(os.path.join("debug", "ocr_input.jpg"), binary)
        logger.debug("Saved OCR input image to debug/ocr_input.jpg")

        # Handle two-line HSRP plates
        height, width = binary.shape
        if height > width * 0.4:  # Likely two-line plate
            logger.debug("Detected possible two-line plate, splitting")
            top_half = binary[:height//2, :]
            bottom_half = binary[height//2:, :]
            cv2.imwrite(os.path.join("debug", "ocr_top_half.jpg"), top_half)
            cv2.imwrite(os.path.join("debug", "ocr_bottom_half.jpg"), bottom_half)
            logger.debug("Saved top and bottom halves for OCR")
        else:
            top_half = binary
            bottom_half = None
            logger.debug("Single-line plate detected")

        psm_modes = [
            (6, "block of text"),
            (7, "single line"),
            (8, "single word"),
            (11, "sparse text")
        ]
        best_text = ""
        best_conf = 0.0
        best_char_confs = []

        for img in [top_half] + ([bottom_half] if bottom_half is not None else []):
            for psm, desc in psm_modes:
                custom_config = (
                    f'--oem 3 --psm {psm} '
                    f'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
                text = ""
                conf = 0.0
                char_confs = []
                for i in range(len(data['text'])):
                    if data['text'][i].strip():
                        text += data['text'][i]
                        conf += float(data['conf'][i])
                        char_confs.append((data['text'][i], data['conf'][i]))
                text = text.strip()
                conf = conf / max(1, len(char_confs)) if char_confs else 0.0
                logger.debug(f"OCR output (psm {psm}, {desc}): '{text}', confidence={conf:.2f}")
                logger.debug(f"Character confidences: {char_confs}")
                if text and conf > best_conf:
                    best_text = text
                    best_conf = conf
                    best_char_confs = char_confs

        if not best_text:
            logger.warning("No text recognized by OCR")
            return ""

        # Character correction for HSRP-specific misreads
        corrected_text = list(best_text)
        for i, (char, conf) in enumerate(best_char_confs):
            if conf < 80:  # Threshold for correction
                if char == 'A' and corrected_text[i] in 'AH':
                    corrected_text[i] = 'H'
                elif char == 'H' and corrected_text[i] in 'HA1':
                    corrected_text[i] = 'H'
                elif char == 'I' and corrected_text[i] in 'I1':
                    corrected_text[i] = '1'
                elif char == '1' and corrected_text[i] in 'I1H':
                    corrected_text[i] = '1'
                elif char == '8' and corrected_text[i] in '89':
                    corrected_text[i] = '9'
                elif char == '9' and corrected_text[i] in '98':
                    corrected_text[i] = '9'
                elif char == 'B' and corrected_text[i] in 'BE':
                    corrected_text[i] = 'E'
                elif char == 'E' and corrected_text[i] in 'EB':
                    corrected_text[i] = 'B'
                elif char == '0' and corrected_text[i] in '09':
                    corrected_text[i] = '9'
                elif char == 'O' and corrected_text[i] in 'O0D':
                    corrected_text[i] = '0'
                elif char == 'D' and corrected_text[i] in 'D0':
                    corrected_text[i] = '0'
                elif char == 'G' and corrected_text[i] in 'GC':
                    corrected_text[i] = 'C'
                elif char == 'C' and corrected_text[i] in 'CG':
                    corrected_text[i] = 'G'
                elif char == 'S' and corrected_text[i] in 'S5':
                    corrected_text[i] = '5'
                elif char == '5' and corrected_text[i] in 'S5':
                    corrected_text[i] = 'S'
                elif char == '2' and corrected_text[i] in '2Z':
                    corrected_text[i] = 'Z'
                elif char == 'Z' and corrected_text[i] in '2Z':
                    corrected_text[i] = '2'
                elif char == '7' and corrected_text[i] in '71':
                    corrected_text[i] = '1'
        corrected_text = ''.join(corrected_text)
        logger.debug(f"Corrected text: '{corrected_text}' (original: '{best_text}', confidence={best_conf:.2f})")

        return corrected_text

    except Exception as e:
        logger.error(f"Error in plate text recognition: {str(e)}", exc_info=True)
        return ""

def clean_plate_text(plate_text):
    """
    Clean and validate HSRP plate text
    """
    try:
        logger.debug(f"Cleaning plate text: '{plate_text}'")
        cleaned = plate_text.replace(" ", "").upper()
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        logger.debug(f"Cleaned text: '{cleaned}'")

        # Validate HSRP format: SSCCVNNNN or SSCCVVNNNN
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', cleaned) and 8 <= len(cleaned) <= 10:
            logger.debug("Text passes HSRP validation")
            return cleaned
        else:
            logger.warning(f"Text validation failed (length={len(cleaned)}, expected HSRP format)")
            return ""

    except Exception as e:
        logger.error(f"Error cleaning plate text: {str(e)}", exc_info=True)
        return ""