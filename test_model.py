import os
import tensorflow as tf
import numpy as np
from predictor import AgeGenderPredictor
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add print statements that flush immediately
def debug_print(message):
    print(message)
    sys.stdout.flush()

def test_prediction_variation():
    """Test if the model predicts different outputs for different inputs"""
    debug_print("Starting test_prediction_variation function...")
    predictor = AgeGenderPredictor()
    debug_print("Loading model...")
    if not predictor.load_model():
        debug_print("Failed to load model")
        return
    
    debug_print("\nTesting if the model produces varied predictions...")
    
    # Create two synthetic test images with different patterns
    # Image 1: Gradient pattern
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            img1[i, j, 0] = i % 255  # R channel
            img1[i, j, 1] = j % 255  # G channel
            img1[i, j, 2] = (i + j) % 255  # B channel
    
    # Image 2: Different pattern
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            img2[i, j, 0] = (255 - i) % 255  # R channel
            img2[i, j, 1] = (255 - j) % 255  # G channel
            img2[i, j, 2] = 128  # B channel
            
    # Image 3: Another pattern
    img3 = np.zeros((200, 200, 3), dtype=np.uint8)
    img3[:100, :, 0] = 255  # Top half red
    img3[100:, :, 1] = 255  # Bottom half green
      # Make predictions
    debug_print("Making predictions on test images...")
    results = []
    for i, img in enumerate([img1, img2, img3], 1):
        debug_print(f"Processing test image {i}...")
        result = predictor.predict(img)
        if result:
            debug_print(f"Test Image {i} Prediction: Age={result['age']}, Gender={result['gender']}")
            results.append(result)
        else:
            debug_print(f"Test Image {i} Prediction failed")
    
    # Check for variation
    if len(results) >= 2:
        ages_same = all(r['age'] == results[0]['age'] for r in results)
        genders_same = all(r['gender'] == results[0]['gender'] for r in results)
        
        if ages_same and genders_same:
            print("\nProblem detected: Model produces identical predictions for different inputs.")
            print("This suggests the model may not be learning the features properly.")
        else:
            print("\nModel produces different predictions for different inputs.")
            if ages_same:
                print("However, all age predictions are identical.")
            if genders_same:
                print("However, all gender predictions are identical.")

if __name__ == "__main__":
    test_prediction_variation()
