import cv2
import numpy as np
import os
import pytesseract
from utils import preprocess_image, decode_netout

# Simplified model loading - we'll implement a mock detection for now
wpod_net = None

print("ANPR module initialized in simplified mode")

def detect_license_plate(img, wpod_net, min_confidence=0.5):
    """
    Detects license plates in an image using the WPOD-NET model
    Returns list of plate images and their coordinates
    """
    if wpod_net is None:
        raise ValueError("WPOD-NET model not loaded")
    
    # Input image dimensions for the network
    input_width, input_height = 240, 80
    
    # Preprocess image
    img_preprocessed = preprocess_image(img)
    
    # Get network prediction
    ratio = float(max(img_preprocessed.shape[:2])) / min(img_preprocessed.shape[:2])
    side = int(ratio * input_width)
    bound_dim = min(side + (side % (2**4)), 2**11)
    
    print("[INFO] Processing image with width: {}, height: {}".format(bound_dim, input_height))
    
    # Create input image for the network
    _, img_scaled = cv2.resize(img_preprocessed, (bound_dim, int(bound_dim / ratio)))
    
    # Input tensor
    net_input = np.expand_dims(img_scaled, 0)
    
    # Run inference
    net_output = wpod_net.predict(net_input)
    
    # Decode output
    detected_plates = decode_netout(net_output[0], min_confidence)
    
    plates = []
    for i, plate in enumerate(detected_plates):
        # Transform coordinates to original image dimensions
        pts = plate.reshape(2, 4)
        pts[0] *= img.shape[1]
        pts[1] *= img.shape[0]
        
        # Extract plate from image
        src_pts = pts.T.astype(np.float32)
        dst_pts = np.array([[0, 0], [input_width, 0], [input_width, input_height], [0, input_height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        plate_img = cv2.warpPerspective(img, M, (input_width, input_height))
        
        # Calculate original image bounding box
        x_coords = pts[0].astype(int)
        y_coords = pts[1].astype(int)
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        bbox = (x1, y1, x2, y2)
        
        # Add to list
        plates.append({
            'img': plate_img,
            'bbox': bbox,
            'confidence': float(detected_plates[i, 8])
        })
    
    return plates

def recognize_plate_text(plate_img):
    """
    Uses OCR to recognize text on a license plate image
    """
    # Preprocess image for OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # OCR configuration
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        text = pytesseract.image_to_string(thresh, config=config).strip()
        return text if text else "UNKNOWN"
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return "ERROR"

def detect_and_recognize_plates(img):
    """
    Main function to detect and recognize license plates in an image
    Returns a list of dictionaries with plate text and bounding boxes
    """
    try:
        # Detect license plates
        plates = detect_license_plate(img, wpod_net)
        
        # Recognize text on each plate
        results = []
        for plate in plates:
            text = recognize_plate_text(plate['img'])
            results.append({
                'text': text,
                'bbox': plate['bbox'],
                'confidence': plate['confidence']
            })
        
        return results
    except Exception as e:
        print(f"Error in detect_and_recognize_plates: {str(e)}")
        return []
