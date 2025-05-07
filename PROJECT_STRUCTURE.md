# ANPR System - Project Structure

## Overview
This is an Automatic Number Plate Recognition (ANPR) system designed to detect and recognize license plates from images and video streams with robust authentication and error handling mechanisms. 

## Key Technologies
- Python Flask backend
- Jinja2 templating
- Image processing libraries
- Authentication system
- Error logging and management
- Multiple camera support
- SQLite/PostgreSQL database

## Project Structure

### Core Application Files
- **main.py**: Entry point for the application
- **app.py**: Flask application setup and configuration
- **config.py**: System configuration management
- **models.py**: Database models for vehicles, logs, users, and settings

### Camera and ANPR Processing
- **camera_manager.py**: Manages camera initialization, capture, and switching
- **anpr_processor.py**: License plate recognition and processing
- **mygate_api.py**: Integration with MyGate API for vehicle entry/exit

### Routes/Controllers
- **routes/__init__.py**: Routes package initialization
- **routes/dashboard.py**: Dashboard and main UI routes
- **routes/api.py**: API endpoints for ANPR control and data retrieval
- **routes/auth.py**: Authentication routes

### Templates
- **templates/base.html**: Base template with layout and navigation
- **templates/dashboard.html**: Main dashboard view
- **templates/login.html**: Login page
- **templates/settings.html**: System settings page
- **templates/vehicles.html**: Vehicle management page
- **templates/logs.html**: Log viewing and filtering

### Static Assets
- **static/css/custom.css**: Custom styling
- **static/js/dashboard.js**: Dashboard JavaScript for real-time updates
- **static/images/captures/**: Directory for captured license plate images

### Utility and Helper Files
- **utils.py**: Utility functions used across the application
- **migrate_db.py**: Database migration script
- **generate_sample_logs.py**: Script to generate sample data

## Key Features

1. **Authentication System**:
   - User login with password hashing
   - Admin privileges control
   - Token-based authentication option

2. **Multi-Camera Support**:
   - Camera switching UI
   - Configuration for multiple cameras
   - Camera selection dropdown

3. **Dashboard and Monitoring**:
   - Real-time logs display
   - Traffic statistics and charts
   - System status monitoring

4. **Vehicle Management**:
   - Registration of resident and visitor vehicles
   - Vehicle search and filtering
   - License plate tracking

5. **Logging and Reporting**:
   - Entry/exit event logging
   - Image storage of captured plates
   - Confidence level tracking

6. **System Configuration**:
   - Society name customization
   - Camera settings management
   - Processing parameters configuration
   - Admin password management

## Installation and Setup

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Initialize the database:
   ```
   python migrate_db.py
   ```

3. Start the application:
   ```
   python main.py
   ```

## Usage
- Access the dashboard at: http://localhost:5000/
- Default admin credentials: admin/admin (change on first login)
- Configure cameras and system settings before starting ANPR processing