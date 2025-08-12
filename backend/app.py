from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import logging
from model_loader import load_model, get_lesion_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://avi-47.github.io",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]}})


MODEL = None 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LESION_INFO = get_lesion_info()

def initialize_model():
    """Initialize the model on startup"""
    import traceback
    global MODEL
    model_path = r'models/supcon_rp_RP_30_v3_temp0.1_best_model.pth'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        MODEL = load_model(model_path, DEVICE)
        logger.info(f"Model loaded successfully on {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
        return False

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor.to(DEVICE)

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('../frontend/static', filename)

@app.route('/health', methods=['GET'])
def health_check():
    print("Health endpoint called")  # Debug
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        try:
            image = Image.open(io.BytesIO(file.read()))
            input_tensor = preprocess_image(image)
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return jsonify({'error': 'Invalid image format'}), 400
        
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        results = []
        for i, prob in enumerate(probabilities):
            lesion = LESION_INFO[i]
            results.append({
                'class_id': i,
                'name': lesion['name'],
                'description': lesion['description'],
                'probability': float(prob),
                'percentage': f"{prob * 100:.1f}%",
                'severity': lesion['severity'],
                'color': lesion['color']
            })
        
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0],
            'device_used': str(DEVICE)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

import os

if __name__ == '__main__':
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        exit(1)

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

