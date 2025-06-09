#!/usr/bin/env python3
"""
Train Age and Gender Prediction CNN Model with Enhanced Reporting
"""
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

from data_loader import UTKFaceDataLoader
from model_architectures.model_architecture import ModelArchitectures
from config_manager.config_manager import ConfigManager
from report_generator import generate_report

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
        self.training_stats = {}
        self.predictions = None
        self.test_data = None
    
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
        """Load and prepare data with enhanced statistics"""
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
        
        # Collect data statistics for reporting
        self.collect_data_statistics()
        
        logger.info(f"Processed {len(self.data_loader.image_paths)} images")
        return True
    
    def collect_data_statistics(self):
        """Collect comprehensive data statistics"""
        ages = [self.data_loader.ages[i] for i in range(len(self.data_loader.ages))]
        genders = [self.data_loader.genders[i] for i in range(len(self.data_loader.genders))]
        
        self.training_stats['data'] = {
            'total_samples': len(ages),
            'age_distribution': {
                'mean': np.mean(ages),
                'std': np.std(ages),
                'min': np.min(ages),
                'max': np.max(ages),
                'median': np.median(ages),
                'q25': np.percentile(ages, 25),
                'q75': np.percentile(ages, 75)
            },
            'gender_distribution': {
                'male_count': sum(genders),
                'female_count': len(genders) - sum(genders),
                'male_percentage': (sum(genders) / len(genders)) * 100,
                'female_percentage': ((len(genders) - sum(genders)) / len(genders)) * 100
            },
            'age_groups': self.get_age_group_distribution(ages),
            'ages': ages,
            'genders': genders
        }
    
    def get_age_group_distribution(self, ages):
        """Get age group distribution"""
        age_groups = {
            '0-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, 
            '41-50': 0, '51-60': 0, '61-70': 0, '71+': 0
        }
        
        for age in ages:
            if age <= 10:
                age_groups['0-10'] += 1
            elif age <= 20:
                age_groups['11-20'] += 1
            elif age <= 30:
                age_groups['21-30'] += 1
            elif age <= 40:
                age_groups['31-40'] += 1
            elif age <= 50:
                age_groups['41-50'] += 1
            elif age <= 60:
                age_groups['51-60'] += 1
            elif age <= 70:
                age_groups['61-70'] += 1
            else:
                age_groups['71+'] += 1
        
        return age_groups
    
    def build_model(self, architecture, img_size, learning_rate):
        """Build and compile model with enhanced metrics"""
        logger.info(f"Building {architecture} model...")
        
        input_shape = (img_size, img_size, 3)
        self.model = ModelArchitectures.get_model(architecture, input_shape)
        
        if self.model is None:
            return False
        
        # Compile model with additional metrics
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
                'age_output': ['mae', 'mse'],
                'gender_output': ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            }
        )
        
        # Store model architecture info
        self.training_stats['model'] = {
            'architecture': architecture,
            'total_params': self.model.count_params(),
            'trainable_params': sum([keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
            'layers': len(self.model.layers),
            'input_shape': input_shape,
            'learning_rate': learning_rate
        }
        
        logger.info("Model compiled successfully")
        logger.info(f"Model has {self.model.count_params()} parameters")
        
        return True
    
    def train_model(self, params):
        """Train the model with enhanced monitoring"""
        logger.info("Starting model training...")
        
        # Create TensorFlow datasets
        train_ds, val_ds, test_ds = self.data_loader.get_tf_datasets()
        if train_ds is None:
            return False
        
        # Store training start time
        training_start = datetime.now()
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),            keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate training time
        training_end = datetime.now()
        training_duration = training_end - training_start
        
        # Store training statistics
        self.training_stats['training'] = {
            'start_time': training_start,
            'end_time': training_end,
            'duration': str(training_duration),
            'epochs_completed': len(self.history.history['loss']),
            'best_epoch': np.argmin(self.history.history['val_loss']) + 1,
            'best_val_loss': np.min(self.history.history['val_loss']),
            'final_train_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1]
        }
        
        logger.info("Training completed")
        return True
    
    def evaluate_model(self, params):
        """Enhanced model evaluation with detailed metrics"""
        logger.info("Evaluating model...")
        
        # Get the test dataset
        _, _, test_ds = self.data_loader.get_tf_datasets()
        
        # Evaluate
        test_results = self.model.evaluate(test_ds, verbose=0)
        
        # Get predictions for detailed analysis
        self.predictions = self.model.predict(test_ds, verbose=0)
          # Get actual test labels
        test_ages = []
        test_genders = []
        for batch in test_ds:
            # Check the structure of batch[1] - could be a tuple or a dict
            if isinstance(batch[1], tuple):
                batch_ages = batch[1][0].numpy()  # First output is age
                batch_genders = batch[1][1].numpy()  # Second output is gender
            elif isinstance(batch[1], dict):
                batch_ages = batch[1]['age_output'].numpy()
                batch_genders = batch[1]['gender_output'].numpy()
            else:
                logger.error(f"Unknown batch format: {type(batch[1])}")
                raise TypeError(f"Cannot handle batch format: {type(batch[1])}")
            
            test_ages.extend(batch_ages)
            test_genders.extend(batch_genders)
        
        test_ages = np.array(test_ages)
        test_genders = np.array(test_genders)
          # Calculate detailed metrics
        # The model returns a list of outputs corresponding to ['age_output', 'gender_output']
        if isinstance(self.predictions, list):
            age_predictions = self.predictions[0].flatten()
            gender_predictions = (self.predictions[1] > 0.5).astype(int).flatten()
        elif isinstance(self.predictions, dict):
            age_predictions = self.predictions['age_output'].flatten()
            gender_predictions = (self.predictions['gender_output'] > 0.5).astype(int).flatten()
        else:
            logger.error(f"Unknown prediction format: {type(self.predictions)}")
            raise TypeError(f"Cannot handle prediction format: {type(self.predictions)}")
        
        # Age metrics
        age_mae = np.mean(np.abs(age_predictions - test_ages))
        age_mse = np.mean((age_predictions - test_ages) ** 2)
        age_rmse = np.sqrt(age_mse)
        age_r2 = 1 - (np.sum((test_ages - age_predictions) ** 2) / 
                      np.sum((test_ages - np.mean(test_ages)) ** 2))
        
        # Gender metrics
        gender_accuracy = np.mean(gender_predictions == test_genders.flatten())
        
        # Confusion matrix for gender
        from sklearn.metrics import confusion_matrix, classification_report
        gender_cm = confusion_matrix(test_genders.flatten(), gender_predictions)
        
        results = {
            'test_loss': test_results[0],
            'age_loss': test_results[1],
            'gender_loss': test_results[2],
            'age_mae': age_mae,
            'age_mse': age_mse,
            'age_rmse': age_rmse,
            'age_r2': age_r2,
            'gender_accuracy': gender_accuracy,
            'gender_confusion_matrix': gender_cm,
            'predictions': {
                'age_pred': age_predictions,
                'gender_pred': gender_predictions,
                'age_true': test_ages,
                'gender_true': test_genders.flatten()
            }
        }
        
        logger.info(f"Test Results:")
        logger.info(f"  Age MAE: {age_mae:.2f} years")
        logger.info(f"  Age RMSE: {age_rmse:.2f} years")
        logger.info(f"  Age R²: {age_r2:.3f}")
        logger.info(f"  Gender Accuracy: {gender_accuracy:.3f}")
        
        return results
    def save_model(self, params, results):
        """Save trained model and update configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"age_gender_model_{params['architecture']}_{timestamp}"
        model_path = os.path.join("models", f"{model_name}.h5")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Save model
        self.model.save(model_path, save_format='h5')
        logger.info(f"Model saved to {model_path}")
        
        # Update configuration
        version = self.config_manager.add_model(params, model_path, results)
        logger.info(f"Model registered as version {version}")
        
        return model_path, version
    
    def run_training_pipeline(self):
        """Run the complete training pipeline with enhanced reporting"""
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
            
            # Generate comprehensive report
            report_dir, pdf_path = generate_report(self, params, results, model_path, version)
            
            logger.info("Training pipeline completed successfully.")
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Model Performance:")
            print(f"   Age MAE: {results['age_mae']:.2f} years")
            print(f"   Gender Accuracy: {results['gender_accuracy']:.1%}")
            print(f"💾 Model saved to: {model_path}")
            print(f"📁 Training report: {report_dir}")
            print(f"📄 PDF report: {pdf_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            print(f"\n❌ Error during training: {str(e)}")
            return False


def main():
    """Main function to run the training process"""
    print("🚀 Starting Age and Gender Prediction Model Training")
    print("📈 Enhanced with comprehensive reporting and analysis")
    trainer = ModelTrainer()
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()