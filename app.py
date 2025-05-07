import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import cv2
import numpy as np

from forms import LoginForm, RegistrationForm
from models import db, User, PlateDetection

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///anpr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
    
    # Save detection to database if we have plates
    if results['plates']:
        detection = PlateDetection(
            plate_text=results['plates'][0],
            confidence=0.95,  # Mock confidence value
            original_image=results['original_image'],
            detection_image=results['detection_image']
        )
        db.session.add(detection)
        db.session.commit()
    
    return render_template('result.html', 
                           original_image=results['original_image'],
                           detection_image=results['detection_image'],
                           plates=results['plates'])

# Admin routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('admin_login'))
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_for('index') in next_page:
            next_page = url_for('admin_dashboard')
        return redirect(next_page)
    
    return render_template('admin_login.html', form=form)

@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    # Calculate some stats
    total_scans = PlateDetection.query.count()
    
    # Calculate detection rate (% of successful detections)
    detection_rate = 95  # Mock value for now
    
    # Get today's scans
    today = datetime.utcnow().date()
    today_scans = PlateDetection.query.filter(
        PlateDetection.timestamp >= today
    ).count()
    
    # Get recent detections
    recent_detections = PlateDetection.query.order_by(
        PlateDetection.timestamp.desc()
    ).limit(10).all()
    
    # Convert to a format suitable for the template
    detections_list = []
    for detection in recent_detections:
        detections_list.append({
            'id': detection.id,
            'timestamp': detection.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'plate_text': detection.plate_text,
            'confidence': int(detection.confidence * 100),
            'image_url': url_for('static', filename=f'uploads/{detection.detection_image}')
        })
    
    stats = {
        'total_scans': total_scans,
        'detection_rate': detection_rate,
        'today_scans': today_scans
    }
    
    return render_template('admin_dashboard.html', 
                           stats=stats, 
                           recent_detections=detections_list)

# Initialize database
def create_tables():
    with app.app_context():
        db.create_all()
        
        # Create admin user if none exists
        if User.query.filter_by(username='admin').first() is None:
            admin = User(username='admin', email='admin@example.com', is_admin=True)
            admin.set_password('adminpassword')
            db.session.add(admin)
            db.session.commit()
            logger.info('Admin user created')

# Call the initialization function
create_tables()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
