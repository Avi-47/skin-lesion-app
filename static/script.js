// DOM elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imageContainer = document.getElementById('imageContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// ✅ Hugging Face hosted API (backend and frontend are same domain)
const API_BASE_URL = '';
const MAX_FILE_SIZE = 10 * 1024 * 1024;

uploadArea.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
imageInput.addEventListener('change', handleImageSelect);
analyzeBtn.addEventListener('click', analyzeImage);

document.addEventListener('DOMContentLoaded', initApp);

async function initApp() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.model_loaded) {
            showError('Model not loaded on server. Please check backend.');
        }
    } catch (error) {
        console.warn('Could not connect to backend:', error);
        showError('Could not connect to backend server. Please ensure it\'s running.');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG, etc.).');
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showError('File too large. Please select an image smaller than 10MB.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        imageContainer.classList.add('active');
        resetResults();
    };
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!imagePreview.src) {
        showError('Please select an image first.');
        return;
    }

    setLoadingState(true);

    try {
        const formData = new FormData();
        const file = imageInput.files[0];
        formData.append('image', file);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
    } finally {
        setLoadingState(false);
    }
}

function displayResults(data) {
    if (!data.success || !data.predictions) {
        showError('Invalid response from server.');
        return;
    }

    const predictions = data.predictions.slice(0, 3);
    
    let resultsHTML = `
        <h3>🎯 Analysis Results</h3>
        <div style="margin-bottom: 1.5rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
            <strong>Top Prediction:</strong> ${data.top_prediction.name} 
            (${data.top_prediction.percentage} confidence)
        </div>
    `;
    
    predictions.forEach((result, index) => {
        const rankEmoji = index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉';
        const percentage = (result.probability * 100).toFixed(1);
        
        resultsHTML += `
            <div class="result-item" style="border-left-color: ${result.color};">
                <div class="lesion-name">
                    ${rankEmoji} ${result.name}
                </div>
                <div class="confidence">
                    Confidence: ${percentage}%
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentage}%; background-color: ${result.color};"></div>
                </div>
                <div class="description">
                    ${result.description}
                </div>
                <span class="severity-badge severity-${result.severity}">
                    ${result.severity} risk
                </span>
            </div>
        `;
    });

    results.innerHTML = resultsHTML;
    results.classList.add('active');
    
    setTimeout(() => {
        const progressBars = document.querySelectorAll('.progress-fill');
        progressBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    }, 100);
}

function setLoadingState(isLoading) {
    if (isLoading) {
        loading.classList.add('active');
        analyzeBtn.disabled = true;
        results.classList.remove('active');
    } else {
        loading.classList.remove('active');
        analyzeBtn.disabled = false;
    }
}

function resetResults() {
    results.classList.remove('active');
}

function showError(message) {
    alert(`Error: ${message}`);
    console.error('Application error:', message);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

document.addEventListener('DOMContentLoaded', function() {
    const container = document.querySelector('.container');
    container.style.opacity = '0';
    container.style.transform = 'translateY(30px)';
    
    setTimeout(() => {
        container.style.transition = 'all 0.8s ease';
        container.style.opacity = '1';
        container.style.transform = 'translateY(0)';
    }, 200);
});
