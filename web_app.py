#!/usr/bin/env python3
"""
Flask Web Application for Age and Gender Prediction
"""
import os
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

from predictor import AgeGenderPredictor
from config_manager.config_manager import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'age-gender-prediction-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global predictor instance
predictor = AgeGenderPredictor()
config_manager = ConfigManager()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_for_display(image_path, max_size=400):
    """Resize image for web display while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Calculate new size maintaining aspect ratio
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save resized image
            resized_path = image_path.replace('.', '_resized.')
            img.save(resized_path, optimize=True, quality=85)
            return resized_path
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image_path

@app.route('/')
def index():
    """Main page"""
    try:
        # Get available models
        available_models = predictor.get_available_models()
        return render_template('index.html', models=available_models)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('index.html', models=[], error="Error loading models")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction from web form"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return render_template('index.html', 
                                 models=predictor.get_available_models(),
                                 error="No image file uploaded")
        
        file = request.files['image']
        model_version = request.form.get('model_version')
        
        if file.filename == '':
            return render_template('index.html',
                                 models=predictor.get_available_models(),
                                 error="No image file selected")
        
        if not allowed_file(file.filename):
            return render_template('index.html',
                                 models=predictor.get_available_models(),
                                 error="Invalid file format. Please upload PNG, JPG, JPEG, GIF, or BMP files.")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load model if not already loaded or different version
        if model_version and model_version.isdigit():
            model_version = int(model_version)
            if predictor.current_version != model_version:
                if not predictor.load_model(model_version):
                    return render_template('index.html',
                                         models=predictor.get_available_models(),
                                         error=f"Failed to load model version {model_version}")
        else:
            if predictor.current_model is None:
                if not predictor.load_model():
                    return render_template('index.html',
                                         models=predictor.get_available_models(),
                                         error="No trained models available")
        
        # Make prediction
        result = predictor.predict(filepath)
        
        if result is None:
            os.remove(filepath)  # Clean up uploaded file
            return render_template('index.html',
                                 models=predictor.get_available_models(),
                                 error="Prediction failed. Please try with a different image.")
          # Resize image for display
        display_image = resize_image_for_display(filepath)
        display_filename = os.path.basename(filepath)
        
        return render_template('index.html',
                             result=result,
                             filename=display_filename,
                             models=predictor.get_available_models())
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template('index.html',
                             models=predictor.get_available_models(),
                             error=f"An error occurred during prediction: {str(e)}")

# REST API Endpoints

@app.route('/api/models', methods=['GET'])
def api_get_models():
    """API endpoint to get available models"""
    try:
        models = predictor.get_available_models()
        return jsonify({
            'success': True,
            'models': [{'version': v, 'created_at': c, 'architecture': a} 
                      for v, c, a in models]
        })
    except Exception as e:
        logger.error(f"API error getting models: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        model_version = request.form.get('model_version')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file format'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load model
            if model_version and model_version.isdigit():
                model_version = int(model_version)
                if predictor.current_version != model_version:
                    if not predictor.load_model(model_version):
                        return jsonify({'success': False, 'error': f'Failed to load model version {model_version}'}), 400
            else:
                if predictor.current_model is None:
                    if not predictor.load_model():
                        return jsonify({'success': False, 'error': 'No trained models available'}), 400
            
            # Make prediction
            result = predictor.predict(filepath)
            
            if result is None:
                return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
            return jsonify({
                'success': True,
                'prediction': {
                    'age': result['age'],
                    'gender': result['gender'],
                    'gender_confidence': result['gender_confidence'],
                    'model_version': result['model_version']
                }
            })
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """API health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_available': len(predictor.get_available_models())
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return render_template('index.html',
                         models=predictor.get_available_models(),
                         error="File too large. Please upload an image smaller than 16MB."), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html',
                         models=predictor.get_available_models(),
                         error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}")
    return render_template('index.html',
                         models=predictor.get_available_models(),
                         error="Internal server error. Please try again."), 500

if __name__ == '__main__':
    # Create templates directory and files if they don't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Age and Gender Prediction Web Application...")
    print("Available endpoints:")
    print("  Web Interface: http://localhost:5000/")
    print("  API Health: http://localhost:5000/api/health")
    print("  API Models: http://localhost:5000/api/models")
    print("  API Predict: http://localhost:5000/api/predict (POST)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)