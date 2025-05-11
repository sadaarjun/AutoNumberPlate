"""
ANPR (Automatic Number Plate Recognition) Processor
Handles all ANPR-related processing using NCNN for inference
"""
import os
import cv2
import numpy as np
import logging
from utils import ensure_directory_exists

# Import ncnn with error handling
try:
    import ncnn
    # Different versions of ncnn have different structures
    # Some versions have different pixel type structures
    
    # Set default
    NEEDS_COLOR_CONVERSION = False
    
    # Try to find the correct pixel type constant
    try:
        # Check for Mat.PixelType structure (newer versions)
        if hasattr(ncnn, 'Mat') and hasattr(ncnn.Mat, 'PixelType'):
            if hasattr(ncnn.Mat.PixelType, 'BGR'):
                PIXEL_BGR = ncnn.Mat.PixelType.BGR
                logging.info("Using ncnn.Mat.PixelType.BGR")
            elif hasattr(ncnn.Mat.PixelType, 'PIXEL_BGR'):
                PIXEL_BGR = ncnn.Mat.PixelType.PIXEL_BGR
                logging.info("Using ncnn.Mat.PixelType.PIXEL_BGR")
            elif hasattr(ncnn.Mat.PixelType, 'PIXEL_RGB'):
                PIXEL_BGR = ncnn.Mat.PixelType.PIXEL_RGB
                NEEDS_COLOR_CONVERSION = True
                logging.info("Using ncnn.Mat.PixelType.PIXEL_RGB with color conversion")
            else:
                PIXEL_BGR = 2  # Fallback value
                logging.warning("Using fallback value 2 for Mat.PixelType.BGR")
        # Check for PixelType structure (older versions)
        elif hasattr(ncnn, 'PixelType'):
            if hasattr(ncnn.PixelType, 'BGR'):
                PIXEL_BGR = ncnn.PixelType.BGR
                logging.info("Using ncnn.PixelType.BGR")
            elif hasattr(ncnn.PixelType, 'PIXEL_BGR'):
                PIXEL_BGR = ncnn.PixelType.PIXEL_BGR
                logging.info("Using ncnn.PixelType.PIXEL_BGR")
            elif hasattr(ncnn.PixelType, 'PIXEL_RGB'):
                PIXEL_BGR = ncnn.PixelType.PIXEL_RGB
                NEEDS_COLOR_CONVERSION = True
                logging.info("Using ncnn.PixelType.PIXEL_RGB with color conversion")
            else:
                PIXEL_BGR = 2  # Fallback
                logging.warning("Using fallback value 2 for PixelType.BGR")
        # Check for direct constants
        elif hasattr(ncnn, 'PIXEL_BGR'):
            PIXEL_BGR = ncnn.PIXEL_BGR
            logging.info("Using ncnn.PIXEL_BGR")
        elif hasattr(ncnn, 'PIXEL_RGB'):
            PIXEL_BGR = ncnn.PIXEL_RGB
            NEEDS_COLOR_CONVERSION = True
            logging.info("Using ncnn.PIXEL_RGB with color conversion")
        else:
            # Fallback to common value
            PIXEL_BGR = 2  # Common value for BGR in some libraries
            logging.warning("Using fallback value 2 for PIXEL_BGR")
    except Exception as e:
        PIXEL_BGR = 2  # Fallback on any error
        logging.warning(f"Error detecting pixel type: {str(e)}. Using fallback value 2.")
except ImportError:
    logging.error("NCNN library not found. Please install it with pip install ncnn")
    raise

logger = logging.getLogger(__name__)

class ANPRProcessor:
    """
    Process images using YOLO v11 for Automatic Number Plate Recognition
    Optimized for Raspberry Pi deployment
    """
    def _is_raspberry_pi(self):
        """Check if we're running on a Raspberry Pi for optimizations"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Raspberry Pi' in model
        except:
            # Check for ARM processor as a fallback
            try:
                import platform
                return platform.machine().startswith('arm') or platform.machine().startswith('aarch')
            except:
                return False
            
    def __init__(self, model_path="models/anpr_yolov11.param", weights_path="models/anpr_yolov11.bin"):
        """
        Initialize the ANPR processor
        
        Args:
            model_path: Path to the NCNN model .param file
            weights_path: Path to the NCNN model .bin file
        """
        self.model_path = model_path
        self.weights_path = weights_path
        
        # Check if running on Raspberry Pi for optimizations
        self.is_raspi = self._is_raspberry_pi()
        
        # Use a smaller input size on Raspberry Pi to improve performance
        if self.is_raspi:
            logger.info("Optimizing ANPR for Raspberry Pi - using reduced inference size")
            self.input_size = (320, 320)  # Smaller input size for better performance on Raspberry Pi
        else:
            self.input_size = (640, 640)  # Default YOLO v11 input size for desktop systems
            
        self.net = None
        self.initialized = False
        
        # Load the model if files exist
        if os.path.exists(model_path) and os.path.exists(weights_path):
            self.load_model()
        else:
            logger.warning(f"Model files not found at {model_path} and {weights_path}")
            logger.info("The model needs to be trained first or copied to the specified location")

    def load_model(self):
        """Load the YOLO v11 model using NCNN with Raspberry Pi optimizations"""
        try:
            # Check if files exist first
            if not os.path.exists(self.model_path):
                logger.error(f"Model parameter file not found: {self.model_path}")
                self.initialized = False
                return
            
            if not os.path.exists(self.weights_path):
                logger.error(f"Model weights file not found: {self.weights_path}")
                self.initialized = False
                return
            
            # Create Net object with error handling
            try:
                self.net = ncnn.Net()
                
                # Apply Raspberry Pi specific optimizations if needed
                if self.is_raspi:
                    logger.info("Applying Raspberry Pi specific NCNN optimizations")
                    
                    # Set the number of threads based on CPU cores (Raspberry Pi typically has 4 cores)
                    try:
                        import multiprocessing
                        num_threads = min(2, multiprocessing.cpu_count())  # Use max 2 threads on Raspberry Pi
                        self.net.set_num_threads(num_threads)
                        logger.info(f"Set NCNN to use {num_threads} threads for inference")
                    except Exception as e:
                        logger.warning(f"Failed to set thread count: {str(e)}")
                    
                    # Enable NCNN optimization options for ARM devices
                    try:
                        # Enable Winograd optimization for faster convolution on ARM
                        if hasattr(self.net, 'opt'):
                            if hasattr(self.net.opt, 'use_winograd_convolution'):
                                self.net.opt.use_winograd_convolution = True
                                
                            # Limit memory usage on Raspberry Pi
                            if hasattr(self.net.opt, 'blob_allocator_strategy'):
                                self.net.opt.blob_allocator_strategy = 0  # Simple memory strategy
                                
                            logger.info("Applied ARM-specific NCNN optimizations")
                    except Exception as e:
                        logger.warning(f"Failed to set ARM optimizations: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to create NCNN Net object: {str(e)}")
                self.initialized = False
                return
                
            # Load parameters and weights
            try:
                ret_param = self.net.load_param(self.model_path)
                if ret_param != 0:
                    logger.error(f"Failed to load model parameters, error code: {ret_param}")
                    self.initialized = False
                    return
                    
                ret_model = self.net.load_model(self.weights_path)
                if ret_model != 0:
                    logger.error(f"Failed to load model weights, error code: {ret_model}")
                    self.initialized = False
                    return
                    
                self.initialized = True
                logger.info("ANPR model loaded successfully")
                
                # For Raspberry Pi, print memory usage to help with debugging
                if self.is_raspi:
                    try:
                        import psutil
                        mem = psutil.virtual_memory()
                        logger.info(f"Memory usage after model loading: {mem.percent}% used, {mem.available / 1024 / 1024:.1f} MB available")
                    except ImportError:
                        logger.info("psutil not available for memory tracking")
                    
            except Exception as e:
                logger.error(f"Failed to load ANPR model: {str(e)}")
                self.initialized = False
        except Exception as e:
            logger.error(f"Unexpected error loading ANPR model: {str(e)}")
            self.initialized = False

    def preprocess_image(self, img):
        """
        Preprocess image for YOLO v11 inference
        Optimized for Raspberry Pi with memory considerations
        
        Args:
            img: OpenCV image in BGR format
            
        Returns:
            Preprocessed image and scaling factors
        """
        # Get original dimensions
        height, width, _ = img.shape
        
        # Calculate scaling factors
        scale_x = self.input_size[0] / width
        scale_y = self.input_size[1] / height
        
        # For Raspberry Pi, use a more memory-efficient resizing method when needed
        if self.is_raspi and (width > 1000 or height > 1000):
            # For large images on Raspberry Pi, do resize in two steps to save memory
            # First resize to an intermediate size
            interim_size = (min(width, 640), min(height, 640))
            img = cv2.resize(img, interim_size)
            
            # Release memory explicitly (helpful on Raspberry Pi)
            import gc
            gc.collect()
            
            # Then resize to final input size
            resized = cv2.resize(img, self.input_size)
        else:
            # Standard resize for regular images
            resized = cv2.resize(img, self.input_size)
        
        # Handle color conversion if needed due to ncnn.PixelType fix
        if globals().get('NEEDS_COLOR_CONVERSION', False):
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize pixel values to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized, (scale_x, scale_y)
    
    def detect(self, img, conf_threshold=0.25, iou_threshold=0.45):
        """
        Detect license plates in the image
        Optimized for performance on Raspberry Pi
        
        Args:
            img: OpenCV image in BGR format
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for non-maximum suppression
            
        Returns:
            List of detected license plates with coordinates and confidence
        """
        if not self.initialized:
            logger.error("Model not initialized. Cannot perform detection.")
            return []
        
        # Adjust confidence threshold on Raspberry Pi to improve performance
        if self.is_raspi:
            # Use a slightly higher confidence threshold on Raspberry Pi
            # to reduce false positives and post-processing workload
            conf_threshold = max(conf_threshold, 0.35)
            
        # Original image dimensions
        height, width, _ = img.shape
        
        # Record start time for performance monitoring on Raspberry Pi
        start_time = None
        if self.is_raspi:
            import time
            start_time = time.time()
            
        # Preprocess the image
        preprocessed, scales = self.preprocess_image(img)
        
        try:
            # Check if net is initialized properly
            if not self.initialized or self.net is None:
                logger.error("Model not initialized. Call load_model() first.")
                return []
                
            # Create ncnn extractor
            try:
                ex = self.net.create_extractor()
            except Exception as e:
                logger.error(f"Failed to create extractor: {str(e)}")
                return []
            
            # Create ncnn Mat from the preprocessed image
            try:
                # If conversion is needed (RGB instead of BGR)
                if NEEDS_COLOR_CONVERSION:
                    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                
                mat_in = ncnn.Mat.from_pixels(
                    preprocessed, 
                    PIXEL_BGR,  # Use our constant instead of ncnn.PixelType.BGR
                    self.input_size[0], 
                    self.input_size[1]
                )
            except Exception as e:
                logger.error(f"Failed to create Mat from pixels: {str(e)}")
                return []
            
            # Normalize
            mean_vals = [0, 0, 0]
            norm_vals = [1/255.0, 1/255.0, 1/255.0]
            mat_in.substract_mean_normalize(mean_vals, norm_vals)
            
            # Input to the model
            ex.input("input", mat_in)
            
            # Get output
            ret, mat_out = ex.extract("output")
            
            # Process outputs
            detections = []
            if ret == 0:
                # Each detection has 7 elements: batch, class, confidence, x1, y1, x2, y2
                for i in range(mat_out.h):
                    values = [mat_out.row(i)[j] for j in range(mat_out.w)]
                    
                    class_id = int(values[1])
                    confidence = values[2]
                    
                    if confidence >= conf_threshold:
                        # Adjust coordinates back to original image dimensions
                        x1 = int(values[3] / scales[0])
                        y1 = int(values[4] / scales[1])
                        x2 = int(values[5] / scales[0])
                        y2 = int(values[6] / scales[1])
                        
                        # Ensure coordinates are within image boundaries
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        # Create detection object
                        detection = {
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
            
            # Apply non-maximum suppression
            result = self._apply_nms(detections, iou_threshold)
            
            # Log performance metrics for Raspberry Pi
            if self.is_raspi and start_time is not None:
                import time
                elapsed = time.time() - start_time
                logger.info(f"ANPR detection completed in {elapsed:.3f} seconds on Raspberry Pi")
                logger.info(f"Detected {len(result)} license plates with confidence threshold {conf_threshold}")
                
                # Log memory usage if psutil is available
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    logger.info(f"Memory usage after detection: {mem.percent}% used, {mem.available / 1024 / 1024:.1f} MB available")
                    
                    # If memory is running low, force garbage collection
                    if mem.percent > 80:
                        import gc
                        gc.collect()
                        logger.warning(f"Memory usage high ({mem.percent}%), performed garbage collection")
                except ImportError:
                    pass
                
            return result
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            # On Raspberry Pi, perform memory cleanup after an error
            if self.is_raspi:
                try:
                    import gc
                    gc.collect()
                except:
                    pass
            return []
    
    def _apply_nms(self, detections, iou_threshold):
        """
        Apply non-maximum suppression to the detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IOU threshold for NMS
            
        Returns:
            List of filtered detections
        """
        if not detections:
            return []
            
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        kept_detections = []
        
        while detections:
            best = detections.pop(0)
            kept_detections.append(best)
            
            # Filter out overlapping boxes
            filtered_detections = []
            for det in detections:
                if self._calculate_iou(best['bbox'], det['bbox']) <= iou_threshold:
                    filtered_detections.append(det)
            
            detections = filtered_detections
            
        return kept_detections
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IOU score
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou

    def save_detections(self, img, detections, output_path="output"):
        """
        Save detection results as annotated images
        
        Args:
            img: Original image
            detections: List of detections
            output_path: Directory to save results
        """
        ensure_directory_exists(output_path)
        
        # Create a copy of the image
        result_img = img.copy()
        
        # Draw detections
        for det in detections:
            box = det['bbox']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_img, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 255, 0), 2)
            
            # Draw label
            label = f"Plate: {confidence:.2f}"
            cv2.putText(result_img, label, 
                       (int(box[0]), int(box[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 2)
        
        # Save the result
        output_file = os.path.join(output_path, "detection_result.jpg")
        cv2.imwrite(output_file, result_img)
        logger.info(f"Detection result saved to {output_file}")
        
        return output_file

    def crop_plates(self, img, detections, output_path="output/plates"):
        """
        Crop detected license plates from the image
        
        Args:
            img: Original image
            detections: List of detections
            output_path: Directory to save cropped plates
            
        Returns:
            List of paths to cropped plate images
        """
        ensure_directory_exists(output_path)
        
        plate_paths = []
        
        for i, det in enumerate(detections):
            box = det['bbox']
            
            # Crop the plate region
            plate_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            
            # Save the cropped plate
            plate_path = os.path.join(output_path, f"plate_{i}.jpg")
            cv2.imwrite(plate_path, plate_img)
            plate_paths.append(plate_path)
            
        logger.info(f"Saved {len(plate_paths)} cropped license plates")
        
        return plate_paths
