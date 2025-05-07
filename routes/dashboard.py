from flask import Blueprint, render_template, redirect, url_for, request, flash
from datetime import datetime
from models import PlateDetection, db
from routes.auth import admin_required, login_required

# Create dashboard blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def index():
    """Main dashboard page"""
    # Calculate some stats
    total_scans = PlateDetection.query.count()
    
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
        'detection_rate': 95,  # Mock value for now
        'today_scans': today_scans
    }
    
    return render_template('admin_dashboard.html', 
                           stats=stats, 
                           recent_detections=detections_list)

@dashboard_bp.route('/admin/detections/<int:detection_id>/delete')
@admin_required
def delete_detection(detection_id):
    """Delete a detection record"""
    detection = PlateDetection.query.get_or_404(detection_id)
    db.session.delete(detection)
    db.session.commit()
    flash('Detection record deleted successfully', 'success')
    return redirect(url_for('dashboard.index'))