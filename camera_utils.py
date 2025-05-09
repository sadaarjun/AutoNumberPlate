import cv2
import logging
import requests
import numpy as np
import os
import subprocess
import re
import time
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def capture_image(camera):
    """
    Capture an image from the camera using its settings

    Args:
        camera: CameraSetting object with connection details

    Returns:
        Tuple (success, result) where:
            - success is a boolean indicating if the capture was successful
            - result is either the captured frame (on success) or an error message (on failure)
    """
    if not camera.url:
        return False, "Camera URL is not set"

    try:
        # Special handling for Raspberry Pi camera module
        if camera.url.lower() in ['picamera', 'raspi', 'rpi', 'pi']:
            return capture_from_picamera()

        # Parse URL to determine camera type
        parsed_url = urlparse(camera.url)

        # For RTSP streams
        if parsed_url.scheme == 'rtsp':
            return capture_from_rtsp(camera)

        # For HTTP/HTTPS image URLs (JPEG/MJPEG)
        elif parsed_url.scheme in ['http', 'https']:
            if camera.url.lower().endswith(('.jpg', '.jpeg', '.png')):
                return capture_from_http_image(camera)
            else:
                # Try MJPEG stream
                return capture_from_mjpeg(camera)

        # For local cameras (0, 1, 2, etc.)
        elif camera.url.isdigit():
            return capture_from_local(camera)

        else:
            return False, f"Unsupported camera URL format: {camera.url}"

    except Exception as e:
        logger.error(f"Error capturing from camera {camera.name}: {str(e)}")
        return False, f"Error: {str(e)}"

def capture_from_rtsp(camera):
    """Capture from RTSP stream"""
    try:
        # Build the RTSP URL with authentication if provided
        url = camera.url
        if camera.username and camera.password and '@' not in url:
            # Insert credentials into URL if not already there
            parsed = urlparse(url)
            url = f"{parsed.scheme}://{camera.username}:{camera.password}@{parsed.netloc}{parsed.path}?{parsed.query}"

        # Open the RTSP stream
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return False, "Failed to open RTSP stream"

        # Read a frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, "Failed to read frame from RTSP stream"

        return True, frame

    except Exception as e:
        return False, f"RTSP Error: {str(e)}"

def capture_from_http_image(camera):
    """Capture from HTTP/HTTPS still image URL"""
    try:
        # Make request with authentication if provided
        auth = None
        if camera.username and camera.password:
            auth = HTTPBasicAuth(camera.username, camera.password)

        # Get the image
        response = requests.get(camera.url, auth=auth, timeout=10)
        if response.status_code != 200:
            return False, f"HTTP Error: {response.status_code}"

        # Convert to image
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return False, "Failed to decode image from HTTP response"

        return True, frame

    except Exception as e:
        return False, f"HTTP Image Error: {str(e)}"

def capture_from_mjpeg(camera):
    """Capture from MJPEG stream"""
    try:
        # Try to open as a video stream
        cap = cv2.VideoCapture(camera.url)
        if not cap.isOpened():
            return False, "Failed to open MJPEG stream"

        # Read a frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, "Failed to read frame from MJPEG stream"

        return True, frame

    except Exception as e:
        return False, f"MJPEG Error: {str(e)}"

def capture_from_local(camera):
    """Capture from local camera (webcam or Raspberry Pi camera)"""
    try:
        # Special handling for Raspberry Pi camera module
        if camera.url.lower() == 'picamera':
            return capture_from_picamera()

        # Open local camera
        cap = cv2.VideoCapture(int(camera.url))
        if not cap.isOpened():
            return False, f"Failed to open local camera {camera.url}"

        # Read a frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, f"Failed to read frame from local camera {camera.url}"

        return True, frame

    except Exception as e:
        return False, f"Local Camera Error: {str(e)}"

def capture_from_picamera():
    """Capture from Raspberry Pi camera module"""
    try:
        # Check if running on Raspberry Pi
        if not is_raspberry_pi():
            return False, "Not running on a Raspberry Pi"

        # Try to use picamera if available
        try:
            # Try to use picamera library first (preferred for Raspberry Pi camera module)
            # This code will only run if picamera is installed
            import importlib.util
            if importlib.util.find_spec("picamera") is not None:
                import picamera
                from picamera.array import PiRGBArray

                # Capture using picamera
                with picamera.PiCamera() as camera:
                    camera.resolution = (1280, 720)
                    camera.framerate = 24
                    # Allow camera to warm up
                    camera.start_preview()
                    time.sleep(2)

                    # Capture a frame
                    raw_capture = PiRGBArray(camera)
                    camera.capture(raw_capture, format="bgr")
                    frame = raw_capture.array

                    return True, frame
            else:
                # If picamera not available, try using raspistill command
                return capture_from_raspistill()

        except ImportError:
            # If picamera import fails, try using raspistill command
            return capture_from_raspistill()

    except Exception as e:
        return False, f"PiCamera Error: {str(e)}"

def capture_from_raspistill():
    """Capture from Raspberry Pi camera using raspistill command"""
    try:
        # Check if running on Raspberry Pi
        if not is_raspberry_pi():
            return False, "Not running on a Raspberry Pi"

        # Create a temporary file to store the image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name

        # Use raspistill command to capture image
        command = ['raspistill', '-o', temp_path, '-t', '1', '-n']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()

        if process.returncode != 0:
            return False, f"raspistill error: {stderr.decode('utf-8')}"

        # Read the captured image
        image = cv2.imread(temp_path)

        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        if image is None:
            return False, "Failed to read captured image from raspistill"

        return True, image

    except Exception as e:
        return False, f"RaspiStill Error: {str(e)}"

def is_raspberry_pi():
    """Check if running on a Raspberry Pi"""
    try:
        # Check for Raspberry Pi model in /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Model'):
                    return 'raspberry pi' in line.lower()
        return False
    except:
        return False
