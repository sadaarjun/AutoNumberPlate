import cv2
import logging
import time
import threading
import os
from picamera2 import Picamera2

class CameraManager:
    def __init__(self, config):
        self.config = config
        self.cameras = {}  # Dictionary to store multiple camera instances
        self.active_camera = None  # Reference to the active camera
        self.last_image = None
        self.lock = threading.Lock()
        self.initialized = False
        self.continuous_capture = False
        self.capture_thread = None
        
        # Try to initialize cameras
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize cameras based on configuration"""
        try:
            # Clean up existing cameras first
            self.cleanup()
            
            # Reset cameras dictionary
            self.cameras = {}
            
            # Check if multi-camera is enabled
            if self.config.is_multi_camera_enabled():
                # Initialize all enabled cameras in the list
                camera_list = self.config.get_camera_list()
                active_camera_id = self.config.get_active_camera_id()
                
                for camera_config in camera_list:
                    if camera_config.get('enabled', False):
                        camera_id = camera_config['id']
                        try:
                            # Initialize PiCamera2 for this camera
                            camera = Picamera2()
                            
                            # Get camera resolution from config
                            resolution = self.config.get_camera_resolution()
                            
                            # Configure the camera
                            camera_config_obj = camera.create_still_configuration(
                                main={"size": resolution},
                                lores={"size": (640, 480)},
                                display="lores"
                            )
                            camera.configure(camera_config_obj)
                            
                            # Start the camera
                            camera.start()
                            
                            # Store the camera
                            self.cameras[camera_id] = {
                                'instance': camera,
                                'name': camera_config['name'],
                                'last_image': None
                            }
                            
                            # Set as active camera if it matches the active ID
                            if camera_id == active_camera_id:
                                self.active_camera = camera_id
                                
                            logging.info(f"Camera '{camera_config['name']}' (ID: {camera_id}) initialized")
                            
                        except Exception as camera_err:
                            logging.error(f"Failed to initialize camera '{camera_config['name']}' (ID: {camera_id}): {str(camera_err)}")
                
                # If no active camera was set, use the first available one
                if not self.active_camera and len(self.cameras) > 0:
                    self.active_camera = list(self.cameras.keys())[0]
                    logging.warning(f"No active camera found, using '{self.active_camera}' as default")
            else:
                # Use single camera mode (backwards compatibility)
                try:
                    # Initialize PiCamera2
                    camera = Picamera2()
                    
                    # Get camera resolution from config
                    resolution = self.config.get_camera_resolution()
                    
                    # Configure the camera
                    camera_config = camera.create_still_configuration(
                        main={"size": resolution},
                        lores={"size": (640, 480)},
                        display="lores"
                    )
                    camera.configure(camera_config)
                    
                    # Start the camera
                    camera.start()
                    
                    # Allow the camera to warm up
                    time.sleep(2)
                    
                    # Store the camera under the 'main' ID
                    self.cameras['main'] = {
                        'instance': camera,
                        'name': 'Main Camera',
                        'last_image': None
                    }
                    
                    # Set as the active camera
                    self.active_camera = 'main'
                    
                    logging.info("Default camera initialized successfully")
                    
                except Exception as e:
                    logging.error(f"Failed to initialize default camera: {str(e)}")
            
            # Check if any camera was initialized
            self.initialized = len(self.cameras) > 0 and self.active_camera is not None
            
            # Start continuous capture if enabled in config and cameras are initialized
            if self.initialized and self.config.get_continuous_capture():
                self.start_continuous_capture()
            
            if not self.initialized:
                logging.error("No cameras could be initialized")
                
        except Exception as e:
            self.initialized = False
            logging.error(f"Failed to initialize cameras: {str(e)}")
    
    def capture_image(self, camera_id=None):
        """Capture a single image from the specified or active camera
        
        Args:
            camera_id (str, optional): ID of the camera to use. If None, uses active camera.
            
        Returns:
            numpy.ndarray: The captured image, or None if an error occurred
        """
        with self.lock:
            if not self.initialized:
                logging.warning("Cameras not initialized, attempting to reinitialize")
                self.initialize_camera()
                if not self.initialized:
                    return None
            
            # If no camera_id specified, use the active camera
            if camera_id is None:
                camera_id = self.active_camera
            
            # Ensure the camera exists
            if camera_id not in self.cameras:
                logging.error(f"Camera ID '{camera_id}' not found")
                return None
            
            try:
                camera_data = self.cameras[camera_id]
                camera = camera_data['instance']
                
                # If continuous capture is enabled, return the last captured image
                if self.continuous_capture and camera_data['last_image'] is not None:
                    return camera_data['last_image'].copy()
                
                # Otherwise, capture a new image
                image = camera.capture_array()
                
                # Convert to BGR format if necessary (picamera2 returns RGB)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Apply any image processing if needed
                image = self.process_image(image)
                
                # Store as last image for this camera
                camera_data['last_image'] = image
                
                # If this is the active camera, also update the main last_image
                if camera_id == self.active_camera:
                    self.last_image = image
                
                return image
            
            except Exception as e:
                logging.error(f"Error capturing image from camera '{camera_id}': {str(e)}")
                # Mark the camera as having an issue
                if camera_id in self.cameras:
                    self.cameras[camera_id]['error'] = str(e)
                
                # If this was the active camera and we have other cameras, try to switch
                if camera_id == self.active_camera and len(self.cameras) > 1:
                    # Find another available camera
                    for other_id in self.cameras:
                        if other_id != camera_id:
                            self.active_camera = other_id
                            logging.warning(f"Switched to backup camera '{other_id}' due to error")
                            break
                
                return None
    
    def process_image(self, image):
        """Apply image processing to improve ANPR results"""
        try:
            # Resize if needed
            max_width = self.config.get_max_image_width()
            if max_width > 0 and image.shape[1] > max_width:
                ratio = max_width / image.shape[1]
                new_height = int(image.shape[0] * ratio)
                image = cv2.resize(image, (max_width, new_height))
            
            # Apply any other processing based on config
            if self.config.get_enhance_contrast():
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                # Split the LAB image into L, A, and B channels
                l, a, b = cv2.split(lab)
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                # Merge the enhanced L channel with A and B channels
                updated_lab = cv2.merge((cl, a, b))
                # Convert LAB back to BGR
                image = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
            
            return image
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return image  # Return original image if processing fails
    
    def start_continuous_capture(self):
        """Start continuous image capture in a separate thread"""
        if self.continuous_capture:
            return
        
        self.continuous_capture = True
        self.capture_thread = threading.Thread(target=self._continuous_capture_thread)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logging.info("Started continuous image capture")
    
    def stop_continuous_capture(self):
        """Stop continuous image capture"""
        self.continuous_capture = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        logging.info("Stopped continuous image capture")
    
    def _continuous_capture_thread(self):
        """Thread function for continuous image capture"""
        while self.continuous_capture:
            # Only continue if we have cameras initialized
            if not self.initialized or not self.active_camera:
                time.sleep(0.5)
                continue
                
            try:
                # Get the active camera (using a temporary variable to avoid race conditions)
                current_active_camera = self.active_camera
                
                # Capture images from all enabled cameras, starting with the active one
                cameras_to_process = list(self.cameras.keys())
                # Move active camera to front of the list
                if current_active_camera in cameras_to_process:
                    cameras_to_process.remove(current_active_camera)
                    cameras_to_process.insert(0, current_active_camera)
                
                for camera_id in cameras_to_process:
                    try:
                        with self.lock:
                            if not self.initialized or camera_id not in self.cameras:
                                continue
                                
                            camera_data = self.cameras[camera_id]
                            camera = camera_data['instance']
                            
                            # Capture image from this camera
                            image = camera.capture_array()
                            
                            # Convert to BGR format if necessary
                            if len(image.shape) == 3 and image.shape[2] == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
                            # Process the image
                            image = self.process_image(image)
                            
                            # Update this camera's last image
                            camera_data['last_image'] = image
                            
                            # If this is the active camera, also update the main last_image
                            if camera_id == current_active_camera:
                                self.last_image = image
                    
                    except Exception as camera_err:
                        logging.error(f"Error in continuous capture for camera '{camera_id}': {str(camera_err)}")
                        
                        # Mark the camera as having an issue
                        if camera_id in self.cameras:
                            self.cameras[camera_id]['error'] = str(camera_err)
                        
                        # If this was the active camera, try to switch to another
                        if camera_id == self.active_camera and len(self.cameras) > 1:
                            for other_id in self.cameras:
                                if other_id != camera_id:
                                    self.active_camera = other_id
                                    logging.warning(f"Switched to backup camera '{other_id}' due to error")
                                    break
            
            except Exception as e:
                logging.error(f"Error in continuous capture main loop: {str(e)}")
                # Try to reinitialize if the whole continuous capture fails
                try:
                    self.initialize_camera()
                except:
                    # If reinitialization fails, sleep a bit longer to avoid flooding logs
                    time.sleep(2)
            
            # Sleep for a short period between capture cycles
            time.sleep(0.1)
    
    def set_active_camera(self, camera_id):
        """Set the active camera to use for captures
        
        Args:
            camera_id (str): ID of the camera to set as active
            
        Returns:
            bool: True if the camera was set as active, False otherwise
        """
        with self.lock:
            if not self.initialized:
                logging.warning("Cameras not initialized, cannot set active camera")
                return False
                
            if camera_id not in self.cameras:
                logging.error(f"Camera ID '{camera_id}' not found, cannot set as active")
                return False
                
            # Set the active camera
            self.active_camera = camera_id
            logging.info(f"Set active camera to '{camera_id}'")
            
            # Also update the config
            try:
                self.config.update_setting('cameras.active_camera', camera_id)
            except Exception as e:
                logging.error(f"Error updating active camera in config: {str(e)}")
                
            return True
    
    def get_camera_list(self):
        """Get a list of all available cameras with their status
        
        Returns:
            list: List of camera information dictionaries
        """
        camera_list = []
        
        with self.lock:
            for camera_id, camera_data in self.cameras.items():
                camera_list.append({
                    'id': camera_id,
                    'name': camera_data.get('name', 'Unnamed Camera'),
                    'is_active': camera_id == self.active_camera,
                    'error': camera_data.get('error', None)
                })
                
        return camera_list
    
    def cleanup(self):
        """Clean up camera resources"""
        self.stop_continuous_capture()
        
        with self.lock:
            # Clean up all cameras
            for camera_id, camera_data in list(self.cameras.items()):
                try:
                    camera = camera_data.get('instance')
                    if camera:
                        camera.stop()
                        camera.close()
                except Exception as e:
                    logging.error(f"Error closing camera '{camera_id}': {str(e)}")
            
            # Reset camera dictionary and status
            self.cameras = {}
            self.active_camera = None
            self.last_image = None
            self.initialized = False
