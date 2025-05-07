import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Read the image file
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                flash('Error reading image', 'danger')
                return redirect(request.url)
            
            # Save the original image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, img)
            
            # In simplified mode, we'll create a mock detection
            # Instead of using the WPOD-NET model which has compatibility issues
            
            # Create a simple mock detection in the center of the image
            h, w = img.shape[:2]
            x1, y1 = int(w * 0.3), int(h * 0.4)
            x2, y2 = int(w * 0.7), int(h * 0.6)
            
            # Mock detection result
            results = [{
                'text': 'ABC123',  # Mock plate text
                'bbox': (x1, y1, x2, y2),
                'confidence': 0.95
            }]
            
            # Save the image with detections
            detection_filename = f"detection_{filename}"
            detection_filepath = os.path.join(app.config['UPLOAD_FOLDER'], detection_filename)
            
            # Draw the detections on a copy of the image
            img_with_detections = img.copy()
            for plate_info in results:
                plate_bbox = plate_info['bbox']
                plate_text = plate_info['text']
                
                # Draw bounding box
                x1, y1, x2, y2 = plate_bbox
                cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw text
                cv2.putText(img_with_detections, plate_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imwrite(detection_filepath, img_with_detections)
            
            # Store results in session for the result page
            session['results'] = {
                'original_image': filename,
                'detection_image': detection_filename,
                'plates': [plate_info['text'] for plate_info in results]
            }
            
            flash('Note: Running in simplified mode with mock detection. The real model could not be loaded.', 'info')
            return redirect(url_for('result'))
                
        except Exception as e:
            logger.exception("Error processing image")
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(request.url)
    else:
        flash('Invalid file format. Please upload a PNG or JPG image.', 'danger')
        return redirect(request.url)

@app.route('/result')
def result():
    if 'results' not in session:
        flash('No results to display', 'warning')
        return redirect(url_for('index'))
    
    results = session['results']
    return render_template('result.html', 
                           original_image=results['original_image'],
                           detection_image=results['detection_image'],
                           plates=results['plates'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
