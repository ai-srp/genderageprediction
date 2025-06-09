#!/usr/bin/env python3
"""
Retrain the Age and Gender Prediction model with improved parameters
"""
import os
import logging
from datetime import datetime
from train_model_v1 import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retrain_model():
    """Retrain the model with improved parameters"""
    print("\n=== Retraining Age and Gender Prediction Model ===\n")
    
    # Create a new model trainer
    trainer = ModelTrainer()
    
    # Set improved training parameters
    params = {
        'architecture': 'deep_cnn',  # Using the deep_cnn architecture
        'epochs': 30,               # Increased from 1 to 30 epochs
        'batch_size': 32,           # Standard batch size
        'learning_rate': 0.001,     # Standard learning rate
        'img_size': 96             # Reduced image size to prevent memory issues
    }
    
    print("Training parameters:")
    for param, value in params.items():
        print(f"- {param}: {value}")
    
    print("\nStarting training...")
    # Set up data loader and model
    trainer.setup_for_training(params)
    
    # Train the model
    trainer.train_model()
    
    # Evaluate the model
    trainer.evaluate_model()
    
    # Generate report
    trainer.generate_report()
    
    print("\nTraining completed successfully!")
    print("Check the logs and reports directories for detailed results.")

if __name__ == "__main__":
    retrain_model()
