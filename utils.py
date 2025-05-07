import cv2
import numpy as np

def preprocess_image(img, resize=True):
    """
    Preprocess the image for WPOD-NET model:
    - Convert to RGB if needed
    - Normalize values to [0,1]
    """
    # Make sure the image is in RGB format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Normalize
    img = img / 255.0
    
    return img

def decode_netout(netout, min_confidence=0.5):
    """
    Decode the output of WPOD-NET model to get license plate coordinates
    """
    # Get dimensions
    h, w, _ = netout.shape
    
    # Find potential plate locations (high confidence cells)
    confident_cells = np.where(netout[:,:,8] >= min_confidence)
    
    if len(confident_cells[0]) == 0:
        return np.array([])  # No plates found
    
    # Extract plate coordinates for each confident cell
    plates = []
    for cell_y, cell_x in zip(confident_cells[0], confident_cells[1]):
        # Get the 8 coordinates and confidence
        coords = netout[cell_y, cell_x, :8].reshape(4, 2)
        confidence = netout[cell_y, cell_x, 8]
        
        # Scale coordinates to cell position
        coords[:, 0] = (coords[:, 0] + cell_x) / w
        coords[:, 1] = (coords[:, 1] + cell_y) / h
        
        plate = np.append(coords.flatten(), confidence)
        plates.append(plate)
    
    return np.array(plates)

def draw_box(img, box, color=(0,255,0), thickness=2):
    """
    Draw a box on the image
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def resize_image(img, target_size=None, interp=cv2.INTER_AREA):
    """
    Resize image while maintaining aspect ratio
    """
    if target_size is None:
        return img
    
    h, w = img.shape[:2]
    
    # Calculate target dimensions
    if isinstance(target_size, int):
        # Size is a single number, use as the longer side
        if h > w:
            target_h = target_size
            target_w = int(w * target_h / h)
        else:
            target_w = target_size
            target_h = int(h * target_w / w)
    else:
        # Size is a tuple (width, height)
        target_w, target_h = target_size
    
    # Resize
    return cv2.resize(img, (target_w, target_h), interpolation=interp)
