#!/usr/bin/env python3
"""
Train Age and Gender Prediction CNN Model
"""
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loader import UTKFaceDataLoader
from model_architectures.model_architecture import ModelArchitectures
from config_manager.config_manager import ConfigManager

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_loader = None
        self.model = None
        self.history = None
    
    def get_training_parameters(self):
        """Get training parameters from user input"""
        print("\n=== Age and Gender Prediction Model Training ===")
        print("Available architectures: simple_cnn, deep_cnn, mobilenet")
        
        params = {}
        
        # Architecture
        while True:
            architecture = input("Enter model architecture (default: simple_cnn): ").strip()
            if not architecture:
                architecture = "simple_cnn"
            if architecture in ["simple_cnn", "deep_cnn", "mobilenet"]:
                params['architecture'] = architecture
                break
            print("Invalid architecture. Please choose from: simple_cnn, deep_cnn, mobilenet")
        
        # Epochs
        while True:
            try:
                epochs = input("Enter number of epochs (default: 50): ").strip()
                epochs = int(epochs) if epochs else 50
                if epochs > 0:
                    params['epochs'] = epochs
                    break
                print("Epochs must be positive")
            except ValueError:
                print("Please enter a valid number")
          # Batch size
        while True:
            try:
                batch_size = input("Enter batch size (default: 16): ").strip()
                batch_size = int(batch_size) if batch_size else 16
                if batch_size > 0:
                    params['batch_size'] = batch_size
                    break
                print("Batch size must be positive")
            except ValueError:
                print("Please enter a valid number")
        
        # Learning rate
        while True:
            try:
                lr = input("Enter learning rate (default: 0.001): ").strip()
                lr = float(lr) if lr else 0.001
                if lr > 0:
                    params['learning_rate'] = lr
                    break
                print("Learning rate must be positive")
            except ValueError:
                print("Please enter a valid number")
          # Image size
        while True:
            try:
                print("Note: Smaller image sizes (64-96) help prevent memory issues.")
                img_size = input("Enter image size (default: 96): ").strip()
                img_size = int(img_size) if img_size else 96
                if img_size > 0:
                    params['img_size'] = img_size
                    break
                print("Image size must be positive")
            except ValueError:
                print("Please enter a valid number")
        
        return params
        
    def load_data(self, img_size, batch_size):
        """Load and prepare data"""
        logger.info("Loading data...")
        self.data_loader = UTKFaceDataLoader(img_size=(img_size, img_size), batch_size=batch_size)
        
        if not self.data_loader.load_data():
            logger.error("Failed to load data")
            return False
        
        # Create train/val/test splits
        logger.info("Creating data splits...")
        splits = self.data_loader.get_splits()
        if splits is None:
            logger.error("Failed to create data splits")
            return False
        
        logger.info(f"Processed {len(self.data_loader.image_paths)} images")
        return True
    
    def build_model(self, architecture, img_size, learning_rate):
        """Build and compile model"""
        logger.info(f"Building {architecture} model...")
        
        input_shape = (img_size, img_size, 3)
        self.model = ModelArchitectures.get_model(architecture, input_shape)
        
        if self.model is None:
            return False
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'age_output': 'mse',
                'gender_output': 'binary_crossentropy'
            },
            loss_weights={
                'age_output': 1.0,
                'gender_output': 1.0
            },
            metrics={
                'age_output': ['mae'],
                'gender_output': ['accuracy']
            }
        )
        
        logger.info("Model compiled successfully")
        logger.info(f"Model has {self.model.count_params()} parameters")
        
        return True
    def train_model(self, params):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Create TensorFlow datasets
        train_ds, val_ds, test_ds = self.data_loader.get_tf_datasets()
        if train_ds is None:
            return False
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model with datasets
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return True
    def evaluate_model(self, params):
        """Evaluate model on test data"""
        logger.info("Evaluating model...")
        
        # Get the test dataset
        _, _, test_ds = self.data_loader.get_tf_datasets()
        
        # Evaluate
        test_results = self.model.evaluate(test_ds, verbose=0)
        
        # Extract metrics
        results = {
            'test_loss': test_results[0],
            'age_loss': test_results[1],
            'gender_loss': test_results[2],
            'age_mae': test_results[3],
            'gender_accuracy': test_results[4]
        }
        
        logger.info(f"Test Results:")
        logger.info(f"  Age MAE: {results['age_mae']:.2f} years")
        logger.info(f"  Gender Accuracy: {results['gender_accuracy']:.3f}")
        
        return results
    
    def save_model(self, params, results):
        """Save trained model and update configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"age_gender_model_{params['architecture']}_{timestamp}"
        model_path = os.path.join("models", f"{model_name}.keras")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Update configuration
        version = self.config_manager.add_model(params, model_path, results)
        logger.info(f"Model registered as version {version}")
        
        return model_path, version
    
    def generate_report(self, params, results, model_path, version):
        """Generate training report with plots"""
        logger.info("Generating training report...")
        
        # Create reports directory
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(reports_dir, f"training_report_v{version}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Plot training history
        if self.history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plots
            axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
            axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Age MAE
            axes[0, 1].plot(self.history.history['age_output_mae'], label='Train Age MAE')
            axes[0, 1].plot(self.history.history['val_age_output_mae'], label='Val Age MAE')
            axes[0, 1].set_title('Age Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE (years)')
            axes[0, 1].legend()
            
            # Gender Loss
            axes[1, 0].plot(self.history.history['gender_output_loss'], label='Train Gender Loss')
            axes[1, 0].plot(self.history.history['val_gender_output_loss'], label='Val Gender Loss')
            axes[1, 0].set_title('Gender Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            
            # Gender Accuracy
            axes[1, 1].plot(self.history.history['gender_output_accuracy'], label='Train Gender Accuracy')
            axes[1, 1].plot(self.history.history['val_gender_output_accuracy'], label='Val Gender Accuracy')
            axes[1, 1].set_title('Gender Classification Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plot_path = os.path.join(report_dir, 'training_history.png')
            fig.savefig(plot_path)
            plt.close(fig)
            
            logger.info(f"Training plots saved to {plot_path}")
        
        # Save report summary
        report_path = os.path.join(report_dir, 'report_summary.txt')        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Age and Gender Prediction Model Training Report ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model Version: {version}\n")
            f.write(f"Model Path: {model_path}\n\n")
            
            f.write("=== Parameters ===\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            
            f.write("\n=== Results ===\n")
            f.write(f"Test Loss: {results['test_loss']:.4f}\n")
            f.write(f"Age Loss: {results['age_loss']:.4f}\n")
            f.write(f"Gender Loss: {results['gender_loss']:.4f}\n")
            f.write(f"Age MAE: {results['age_mae']:.2f} years\n")
            f.write(f"Gender Accuracy: {results['gender_accuracy']:.2%}\n")
        
        logger.info(f"Training report saved to {report_dir}")
        return report_dir
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Get training parameters
            params = self.get_training_parameters()
              # Load data
            if not self.load_data(params['img_size'], params['batch_size']):
                logger.error("Data loading failed. Aborting training.")
                return False
            
            # Build model
            if not self.build_model(
                params['architecture'], 
                params['img_size'], 
                params['learning_rate']
            ):
                logger.error("Model building failed. Aborting training.")
                return False
            
            # Train model
            if not self.train_model(params):
                logger.error("Model training failed. Aborting.")
                return False
            
            # Evaluate model
            results = self.evaluate_model(params)
            
            # Save model
            model_path, version = self.save_model(params, results)
            
            # Generate report
            report_dir = self.generate_report(params, results, model_path, version)
            
            logger.info("Training pipeline completed successfully.")
            print(f"\nTraining completed successfully!")
            print(f"Model saved to: {model_path}")
            print(f"Training report: {report_dir}")
            
            return True
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            print(f"\nError during training: {str(e)}")
            return False


def main():
    """Main function to run the training process"""
    print("Starting Age and Gender Prediction Model Training")
    trainer = ModelTrainer()
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()