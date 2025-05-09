document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const cameraButtons = document.querySelectorAll('.camera-select-btn');
    const cameraTestPanel = document.getElementById('cameraTestPanel');
    const selectedCameraTitle = document.getElementById('selectedCameraTitle');
    const captureBtn = document.getElementById('captureBtn');
    const processAnprBtn = document.getElementById('processAnprBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const loadingText = document.getElementById('loadingText');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const captureResultContainer = document.getElementById('captureResultContainer');
    const capturedImage = document.getElementById('capturedImage');
    const anprResultContainer = document.getElementById('anprResultContainer');
    const anprSuccess = document.getElementById('anprSuccess');
    const anprError = document.getElementById('anprError');
    const plateText = document.getElementById('plateText');
    const anprErrorText = document.getElementById('anprErrorText');
    
    // Variables
    let selectedCameraId = null;
    let currentImageData = null;
    
    // Add event listeners to camera buttons
    cameraButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Highlight the selected camera
            cameraButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Get camera ID and name
            selectedCameraId = this.dataset.cameraId;
            const cameraName = this.querySelector('h6').textContent;
            
            // Update the test panel title and show the panel
            selectedCameraTitle.textContent = `Testing: ${cameraName}`;
            cameraTestPanel.style.display = 'block';
            
            // Reset the test panel
            resetTestPanel();
        });
    });
    
    // Capture button click handler
    captureBtn.addEventListener('click', function() {
        if (!selectedCameraId) return;
        
        // Reset previous results
        resetTestPanel();
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        loadingText.textContent = 'Capturing image...';
        
        // Make API request
        fetch('/api/capture_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_id: selectedCameraId
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            if (data.success) {
                // Show the captured image
                currentImageData = data.image;
                capturedImage.src = data.image;
                captureResultContainer.style.display = 'block';
                
                // Enable ANPR processing button
                processAnprBtn.disabled = false;
            } else {
                // Show error message
                showError(data.error || 'Unknown error occurred while capturing image');
            }
        })
        .catch(error => {
            // Hide loading indicator and show error
            loadingIndicator.style.display = 'none';
            showError(`Network error: ${error.message}`);
        });
    });
    
    // Process ANPR button click handler
    processAnprBtn.addEventListener('click', function() {
        if (!currentImageData) return;
        
        // Hide previous ANPR results
        anprResultContainer.style.display = 'none';
        anprSuccess.style.display = 'none';
        anprError.style.display = 'none';
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        loadingText.textContent = 'Processing ANPR...';
        
        // Make API request
        fetch('/api/process_anpr', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_id: selectedCameraId,
                image: currentImageData
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Show ANPR result container
            anprResultContainer.style.display = 'block';
            
            if (data.success) {
                // Show success result
                anprSuccess.style.display = 'block';
                plateText.textContent = data.plate_text;
            } else {
                // Show error result
                anprError.style.display = 'block';
                anprErrorText.textContent = data.error || 'No license plate detected';
            }
        })
        .catch(error => {
            // Hide loading indicator and show error
            loadingIndicator.style.display = 'none';
            showError(`Network error: ${error.message}`);
        });
    });
    
    // Helper function to reset the test panel
    function resetTestPanel() {
        errorMessage.style.display = 'none';
        loadingIndicator.style.display = 'none';
        captureResultContainer.style.display = 'none';
        anprResultContainer.style.display = 'none';
        processAnprBtn.disabled = true;
        currentImageData = null;
    }
    
    // Helper function to show an error message
    function showError(message) {
        errorText.textContent = message;
        errorMessage.style.display = 'block';
    }
});
