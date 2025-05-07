# ANPR System - Technical Documentation

## System Architecture

The ANPR (Automatic Number Plate Recognition) system follows a modular architecture that separates concerns and promotes maintainability. The system consists of the following main components:

### 1. Web Application (Flask)
- **Flask Application**: Handles HTTP requests, routing, and template rendering
- **Templates**: HTML templates with Jinja2 for dynamic rendering
- **Static Assets**: CSS, JavaScript, and images

### 2. Database Layer
- **SQLAlchemy ORM**: Object-Relational Mapping for database interactions
- **Models**: Python classes that represent database tables
- **Database Migrations**: Scripts for managing database schema changes

### 3. Camera Management
- **Camera Manager**: Initializes and manages camera devices
- **Image Capture**: Functions for capturing images from cameras
- **Multi-Camera Support**: Infrastructure for managing multiple cameras

### 4. ANPR Processing
- **License Plate Detection**: Algorithms for detecting license plates in images
- **OCR (Optical Character Recognition)**: Text extraction from license plate images
- **Confidence Scoring**: Evaluation of recognition accuracy

### 5. Authentication and Security
- **User Authentication**: Login, session management, and permissions
- **Password Protection**: Secure password hashing
- **Access Control**: Role-based access to features

## Component Details

### Web Application

The application uses Flask as its web framework. Key aspects include:

**Blueprint Organization:**
- `dashboard_bp`: Main dashboard, vehicle and log management
- `api_bp`: API endpoints for ANPR control and data retrieval
- `auth_bp`: Authentication routes

**Template Structure:**
- Base templates with common layout
- Partial templates for reusable components
- Page-specific templates

**JavaScript Functionality:**
- Real-time updates via AJAX
- Chart visualization of data
- Form handling and validation

### Database Models

**User Model:**
- User authentication information
- Role-based permissions (admin vs. regular user)
- Password hashing

**Vehicle Model:**
- License plate information with proper formatting
- Owner details
- Resident status
- Vehicle type and notes

**Log Model:**
- Entry/exit events
- Timestamps
- Confidence scores
- Image paths
- Processing status

**Setting Model:**
- System configuration storage
- Type-safe setting management
- Centralized configuration

### Camera Management

The `CameraManager` class provides a unified interface for camera operations:

**Initialization:**
- Camera discovery
- Configuration loading
- Error handling for unavailable hardware

**Capture Methods:**
- Single image capture
- Continuous capture in background thread
- Image preprocessing

**Multi-Camera Features:**
- Camera selection
- Active camera tracking
- Camera status monitoring

### ANPR Processing

The `ANPRProcessor` class handles license plate recognition:

**Processing Loop:**
- Image acquisition
- Plate detection
- Result processing

**Integration Points:**
- MyGate API integration for entry/exit events
- Database logging of recognition results
- Image storage of detected plates

**Error Handling:**
- Retry mechanisms
- Graceful degradation
- Error logging

## Configuration System

The configuration system uses a combination of:

**File-Based Configuration:**
- JSON configuration file
- Default values
- Type validation

**Database-Backed Configuration:**
- Runtime settings storage
- UI for configuration changes
- Automatic refresh

**Configuration Categories:**
- Camera settings (resolution, framerate)
- Processing parameters (intervals, thresholds)
- System settings (society name, retention periods)
- Integration settings (API credentials)

## Security Considerations

The system implements several security measures:

**Authentication:**
- Password hashing with Werkzeug's security functions
- Session management
- CSRF protection

**Data Protection:**
- Input validation
- Output escaping
- Parameter sanitization

**Error Handling:**
- Comprehensive try-except blocks
- Informative error messages
- Audit logging

## Installation and Deployment

See README.md for detailed installation instructions. Key deployment considerations:

**Development Environment:**
- Flask development server
- Debug mode enabled
- Live reloading

**Production Environment:**
- Gunicorn WSGI server
- Systemd service for auto-start
- PostgreSQL database

**Environment Configuration:**
- Environment variables for sensitive information
- Config files for persistent settings
- Database connection strings

## Customization and Extension

The system is designed to be customizable and extensible:

**Society Name Customization:**
- Configurable society name displayed throughout the UI
- Stored in database for persistence
- Accessible via config object

**Admin Password Management:**
- Password change form with validation
- Secure hashing of passwords
- Error handling and feedback

**Multi-Camera Configuration:**
- UI for camera selection
- API endpoints for camera operations
- Configuration storage for camera settings

## API Reference

### Internal APIs

**Camera Manager API:**
- `initialize_camera()`: Set up cameras
- `capture_image()`: Capture a single image
- `set_active_camera(camera_id)`: Switch active camera
- `get_camera_list()`: Get list of available cameras

**ANPR Processor API:**
- `start_processing()`: Begin ANPR processing
- `stop_processing()`: Stop ANPR processing
- `recognize_plate(image)`: Process a single image
- `process_vehicle_event(plate_number, confidence, image_path)`: Handle recognized plate

**Configuration API:**
- `load_config()`: Load configuration from files or database
- `update_setting(key, value)`: Update a configuration setting
- `get_setting(key)`: Retrieve a configuration value

### External REST APIs

**ANPR Control Endpoints:**
- `POST /api/anpr/start`: Start ANPR processing
- `POST /api/anpr/stop`: Stop ANPR processing

**Camera Endpoints:**
- `GET /api/camera/list`: Get list of cameras
- `POST /api/camera/switch`: Switch active camera
- `GET /api/camera/capture`: Capture a single image

**Data Retrieval Endpoints:**
- `GET /api/logs/recent`: Get recent log entries
- `GET /api/logs/today`: Get today's logs
- `GET /api/stats`: Get system statistics
- `GET /api/vehicles/search`: Search vehicles

**Settings Endpoint:**
- `POST /api/settings/update`: Update system settings

## Error Handling

The system implements comprehensive error handling:

**Exception Handling:**
- Try-except blocks for all operations
- Specific exception types
- Contextual error messages

**User Feedback:**
- Flash messages for user notification
- Error status codes for API responses
- Friendly error pages

**Logging:**
- Different log levels (INFO, WARNING, ERROR)
- Timestamp and context information
- Log rotation for production

## Performance Considerations

**Database Optimizations:**
- Indexing on frequently queried columns
- Pagination for large result sets
- Query optimization

**Image Processing:**
- Configurable image resolutions
- Optional preprocessing steps
- Caching where appropriate

**Background Processing:**
- Asynchronous processing of ANPR tasks
- Thread safety with locks
- Resource limitation safeguards

## Known Limitations

1. **Platform Dependencies:**
   - Some features require specific hardware (Raspberry Pi Camera)
   - OpenCV dependency may require compilation on some platforms

2. **Authentication Issues:**
   - Token-based authentication has navigation issues on certain pages
   - Session expiration handling needs improvement

3. **Mock Mode Limitations:**
   - In mock mode, camera selection shows placeholder data
   - Recognition confidence is simulated in mock mode

## Future Enhancements

1. **Mobile App Integration:**
   - Develop companion mobile app
   - Push notifications for entry/exit events

2. **Advanced Analytics:**
   - Machine learning for regular visitor detection
   - Traffic pattern analysis

3. **Hardware Integration:**
   - Support for barrier control hardware
   - Integration with additional sensors (motion, infrared)

4. **User Experience:**
   - Enhanced mobile responsiveness
   - Dark mode support
   - Accessibility improvements