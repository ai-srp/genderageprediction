import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from config_manager.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class AgeGenderPredictor:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.current_model = None
        self.current_version = None
        self.img_size = (128, 128)  # Default size
    
    def load_model(self, version=None):
        """Load model by version or latest"""
        try:
            if version is None:
                # Load latest model
                latest_model = self.config_manager.get_latest_model()
                if latest_model is None:
                    logger.error("No trained models found")
                    return False
                model_path = latest_model['model_path']
                self.current_version = latest_model['version']
                # Get image size from model parameters
                if 'img_size' in latest_model['parameters']:
                    size = latest_model['parameters']['img_size']
                    self.img_size = (size, size)
            else:
                # Load specific version
                model_path = self.config_manager.get_model_path(version)
                if model_path is None:
                    logger.error(f"Model version {version} not found")
                    return False
                self.current_version = version
                # Get image size from config
                for model in self.config_manager.config['models']:
                    if model['version'] == version:
                        if 'img_size' in model['parameters']:
                            size = model['parameters']['img_size']
                            self.img_size = (size, size)
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            self.current_model = keras.models.load_model(model_path)
            logger.info(f"Loaded model version {self.current_version} from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    return None
            else:
                # Assume it's already a numpy array
                image = image_path
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """Predict age and gender from image"""
        if self.current_model is None:
            logger.error("No model loaded. Call load_model() first.")
            return None
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None
        
        try:
            # Make prediction
            predictions = self.current_model.predict(processed_image, verbose=0)
            
            # Extract predictions
            age_pred = predictions[0][0][0]  # Age output
            gender_pred = predictions[1][0][0]  # Gender output
            
            # Process predictions
            predicted_age = max(0, int(round(age_pred)))  # Ensure non-negative
            predicted_gender = "Female" if gender_pred > 0.5 else "Male"
            gender_confidence = float(gender_pred if gender_pred > 0.5 else 1 - gender_pred)
            
            result = {
                'age': predicted_age,
                'gender': predicted_gender,
                'gender_confidence': gender_confidence,
                'model_version': self.current_version
            }
            
            logger.info(f"Prediction: Age={predicted_age}, Gender={predicted_gender} ({gender_confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def get_available_models(self):
        """Get list of available models"""
        return self.config_manager.get_available_models()
    
    def predict_batch(self, image_paths):
        """Predict age and gender for multiple images"""
        if self.current_model is None:
            logger.error("No model loaded. Call load_model() first.")
            return None
        
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        
        return results

# Standalone prediction script
def main():
    """Main function for standalone prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict age and gender from image')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--model-version', type=int, help='Model version to use (default: latest)')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    predictor = AgeGenderPredictor()
    
    # List models if requested
    if args.list_models:
        models = predictor.get_available_models()
        if models:
            print("Available models:")
            for version, created_at, architecture in models:
                print(f"  Version {version}: {architecture} (created: {created_at})")
        else:
            print("No trained models found")
        return
    
    # Load model
    if not predictor.load_model(args.model_version):
        print("Failed to load model")
        return
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        return
    
    # Make prediction
    result = predictor.predict(args.image_path)
    if result:
        print(f"\nPrediction Results:")
        print(f"Age: {result['age']} years")
        print(f"Gender: {result['gender']} (confidence: {result['gender_confidence']:.2f})")
        print(f"Model Version: {result['model_version']}")
    else:
        print("Prediction failed")

if __name__ == "__main__":
    main()