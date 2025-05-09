document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const addCameraBtn = document.getElementById('addCameraBtn');
    const newCameraForm = document.getElementById('newCameraForm');
    const cancelAddCamera = document.getElementById('cancelAddCamera');
    const noCamerasMessage = document.getElementById('noCamerasMessage');
    const deleteButtons = document.querySelectorAll('.delete-camera-btn');
    
    // Add camera button click handler
    addCameraBtn.addEventListener('click', function() {
        // Hide no cameras message if visible
        if (noCamerasMessage) {
            noCamerasMessage.style.display = 'none';
        }
        
        // Show the new camera form
        newCameraForm.style.display = 'block';
        
        // Scroll to the form
        newCameraForm.scrollIntoView({ behavior: 'smooth' });
    });
    
    // Cancel add camera button click handler
    cancelAddCamera.addEventListener('click', function() {
        // Hide the new camera form
        newCameraForm.style.display = 'none';
        
        // Show no cameras message if there are no cameras
        if (noCamerasMessage && document.querySelectorAll('.accordion-item').length === 0) {
            noCamerasMessage.style.display = 'block';
        }
    });
    
    // Delete camera buttons click handler
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const cameraId = this.dataset.cameraId;
            const cameraItem = document.getElementById(`camera-item-${cameraId}`);
            
            if (confirm('Are you sure you want to delete this camera?')) {
                // Make API request to delete the camera
                fetch('/api/delete_camera', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        camera_id: cameraId
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Remove the camera item from the DOM
                        cameraItem.remove();
                        
                        // Show no cameras message if there are no more cameras
                        if (document.querySelectorAll('.accordion-item').length === 0 && noCamerasMessage) {
                            noCamerasMessage.style.display = 'block';
                        }
                        
                        // Show success message
                        showAlert('Camera deleted successfully', 'success');
                    } else {
                        // Show error message
                        showAlert(data.error || 'Failed to delete camera', 'danger');
                    }
                })
                .catch(error => {
                    showAlert(`Network error: ${error.message}`, 'danger');
                });
            }
        });
    });
    
    // Helper function to show an alert message
    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert at the top of the content
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }, 5000);
    }
});
