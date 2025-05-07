# Configuration settings for the ANPR system

# Model settings
MODEL_PATH = 'weights/wpod-net.json'
MODEL_WEIGHTS_PATH = 'weights/wpod-net_update.h5'  # If using separate weights file

# Detection settings
DETECTION_THRESHOLD = 0.5  # Minimum confidence for plate detection
INPUT_WIDTH = 240
INPUT_HEIGHT = 80

# OCR settings
OCR_CONFIG = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Application settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
