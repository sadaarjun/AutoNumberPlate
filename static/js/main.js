// Initialize Feather icons
document.addEventListener('DOMContentLoaded', () => {
    feather.replace();
    
    // Preview selected image before upload
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const preview = document.createElement('div');
            preview.id = 'image-preview';
            preview.className = 'mt-3';
            
            // Remove any existing preview
            const existingPreview = document.getElementById('image-preview');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    preview.innerHTML = `
                        <div class="card">
                            <div class="card-header">Image Preview</div>
                            <div class="card-body text-center">
                                <img src="${e.target.result}" class="img-fluid rounded" style="max-height: 300px;" alt="Image Preview">
                            </div>
                        </div>
                    `;
                    
                    fileInput.parentNode.appendChild(preview);
                }
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
    
    // Add spinner to submit button when form is submitted
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitButton.disabled = true;
            }
        });
    }
});
