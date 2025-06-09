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
                'gender_output': ['accuracy', 'precision', 'recall']
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
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.keras',
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
            batch_ages = batch[1]['age_output'].numpy()
            batch_genders = batch[1]['gender_output'].numpy()
            test_ages.extend(batch_ages)
            test_genders.extend(batch_genders)
        
        test_ages = np.array(test_ages)
        test_genders = np.array(test_genders)
        
        # Calculate detailed metrics
        age_predictions = self.predictions[0].flatten()
        gender_predictions = (self.predictions[1] > 0.5).astype(int).flatten()
        
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
    
    def create_comprehensive_plots(self, results, report_dir):
        """Create comprehensive visualization plots"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Training History (Enhanced)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training History and Performance Analysis', fontsize=16, fontweight='bold')
        
        # Loss plots
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Age MAE
        axes[0, 1].plot(self.history.history['age_output_mae'], label='Train Age MAE', linewidth=2)
        axes[0, 1].plot(self.history.history['val_age_output_mae'], label='Val Age MAE', linewidth=2)
        axes[0, 1].set_title('Age Mean Absolute Error', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (years)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gender Accuracy
        axes[0, 2].plot(self.history.history['gender_output_accuracy'], label='Train Gender Accuracy', linewidth=2)
        axes[0, 2].plot(self.history.history['val_gender_output_accuracy'], label='Val Gender Accuracy', linewidth=2)
        axes[0, 2].set_title('Gender Classification Accuracy', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Age Loss Breakdown
        axes[1, 1].plot(self.history.history['age_output_loss'], label='Train Age Loss', linewidth=2)
        axes[1, 1].plot(self.history.history['val_age_output_loss'], label='Val Age Loss', linewidth=2)
        axes[1, 1].set_title('Age Prediction Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Gender Loss Breakdown
        axes[1, 2].plot(self.history.history['gender_output_loss'], label='Train Gender Loss', linewidth=2)
        axes[1, 2].plot(self.history.history['val_gender_output_loss'], label='Val Gender Loss', linewidth=2)
        axes[1, 2].set_title('Gender Classification Loss', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'training_history_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Data Distribution Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Analysis and Feature Distribution', fontsize=16, fontweight='bold')
        
        # Age distribution
        axes[0, 0].hist(self.training_stats['data']['ages'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.training_stats['data']['age_distribution']['mean'], 
                          color='red', linestyle='--', label=f"Mean: {self.training_stats['data']['age_distribution']['mean']:.1f}")
        axes[0, 0].set_title('Age Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gender distribution (pie chart)
        gender_counts = [self.training_stats['data']['gender_distribution']['female_count'],
                        self.training_stats['data']['gender_distribution']['male_count']]
        axes[0, 1].pie(gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%', 
                      colors=['pink', 'lightblue'], startangle=90)
        axes[0, 1].set_title('Gender Distribution', fontweight='bold')
        
        # Age groups bar chart
        age_groups = self.training_stats['data']['age_groups']
        axes[0, 2].bar(age_groups.keys(), age_groups.values(), color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Age Groups Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Age Groups')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Age vs Gender scatter
        male_ages = [age for i, age in enumerate(self.training_stats['data']['ages']) 
                    if self.training_stats['data']['genders'][i] == 1]
        female_ages = [age for i, age in enumerate(self.training_stats['data']['ages']) 
                      if self.training_stats['data']['genders'][i] == 0]
        
        axes[1, 0].hist([female_ages, male_ages], bins=20, alpha=0.7, 
                       label=['Female', 'Male'], color=['pink', 'lightblue'], edgecolor='black')
        axes[1, 0].set_title('Age Distribution by Gender', fontweight='bold')
        axes[1, 0].set_xlabel('Age (years)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot for age by gender
        gender_labels = ['Female' if g == 0 else 'Male' for g in self.training_stats['data']['genders']]
        age_gender_df = pd.DataFrame({'Age': self.training_stats['data']['ages'], 'Gender': gender_labels})
        
        sns.boxplot(data=age_gender_df, x='Gender', y='Age', ax=axes[1, 1])
        axes[1, 1].set_title('Age Distribution Box Plot by Gender', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Data quality indicators
        axes[1, 2].text(0.1, 0.9, f"Total Samples: {self.training_stats['data']['total_samples']}", 
                       transform=axes[1, 2].transAxes, fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.8, f"Age Range: {self.training_stats['data']['age_distribution']['min']:.0f} - {self.training_stats['data']['age_distribution']['max']:.0f} years", 
                       transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].text(0.1, 0.7, f"Age Mean ± Std: {self.training_stats['data']['age_distribution']['mean']:.1f} ± {self.training_stats['data']['age_distribution']['std']:.1f}", 
                       transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].text(0.1, 0.6, f"Gender Balance: {self.training_stats['data']['gender_distribution']['female_percentage']:.1f}% F / {self.training_stats['data']['gender_distribution']['male_percentage']:.1f}% M", 
                       transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].set_title('Dataset Summary', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'data_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Age prediction scatter plot
        age_pred = results['predictions']['age_pred']
        age_true = results['predictions']['age_true']
        
        axes[0, 0].scatter(age_true, age_pred, alpha=0.6, color='blue', s=20)
        axes[0, 0].plot([age_true.min(), age_true.max()], [age_true.min(), age_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Age')
        axes[0, 0].set_ylabel('Predicted Age')
        axes[0, 0].set_title(f'Age Prediction Accuracy\nR² = {results["age_r2"]:.3f}, MAE = {results["age_mae"]:.2f}', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Age prediction residuals
        residuals = age_pred - age_true
        axes[0, 1].scatter(age_pred, residuals, alpha=0.6, color='green', s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Age')
        axes[0, 1].set_ylabel('Residuals (Pred - True)')
        axes[0, 1].set_title('Age Prediction Residuals', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gender confusion matrix
        cm = results['gender_confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
        axes[1, 0].set_title(f'Gender Classification Confusion Matrix\nAccuracy = {results["gender_accuracy"]:.3f}', fontweight='bold')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('True')
        
        # Model architecture summary
        model_info = self.training_stats['model']
        axes[1, 1].text(0.1, 0.9, f"Architecture: {model_info['architecture']}", 
                       transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.8, f"Total Parameters: {model_info['total_params']:,}", 
                       transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.7, f"Trainable Parameters: {model_info['trainable_params']:,}", 
                       transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.6, f"Layers: {model_info['layers']}", 
                       transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.5, f"Input Shape: {model_info['input_shape']}", 
                       transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.4, f"Learning Rate: {model_info['learning_rate']}", 
                       transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.3, f"Training Duration: {self.training_stats['training']['duration']}", 
                       transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].set_title('Model Configuration', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'model_performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comprehensive plots generated successfully")
    
    def generate_pdf_report(self, params, results, model_path, version, report_dir):
        """Generate comprehensive PDF report"""
        pdf_path = os.path.join(report_dir, f'complete_training_report_v{version}.pdf')
        
        with PdfPages(pdf_path) as pdf:
            # Cover Page
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Age and Gender Prediction Model\nTraining Report', 
                        fontsize=24, fontweight='bold', y=0.85)
            
            # Add report metadata
            plt.text(0.5, 0.7, f"Model Version: {version}", ha='center', fontsize=16, transform=fig.transFigure)
            plt.text(0.5, 0.65, f"Architecture: {params['architecture']}", ha='center', fontsize=14, transform=fig.transFigure)
            plt.text(0.5, 0.6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=12, transform=fig.transFigure)
            
            # Add key metrics
            plt.text(0.5, 0.5, "Key Performance Metrics", ha='center', fontsize=18, fontweight='bold', transform=fig.transFigure)
            plt.text(0.5, 0.45, f"Age MAE: {results['age_mae']:.2f} years", ha='center', fontsize=14, transform=fig.transFigure)
            plt.text(0.5, 0.4, f"Age RMSE: {results['age_rmse']:.2f} years", ha='center', fontsize=14, transform=fig.transFigure)
            plt.text(0.5, 0.35, f"Age R²: {results['age_r2']:.3f}", ha='center', fontsize=14, transform=fig.transFigure)
            plt.text(0.5, 0.3, f"Gender Accuracy: {results['gender_accuracy']:.1%}", ha='center', fontsize=14, transform=fig.transFigure)
            
            plt.text(0.5, 0.15, f"Training Duration: {self.training_stats['training']['duration']}", ha='center', fontsize=12, transform=fig.transFigure)
            plt.text(0.5, 0.1, f"Total Parameters: {self.training_stats['model']['total_params']:,}", ha='center', fontsize=12, transform=fig.transFigure)
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 1: Training History
            img_path = os.path.join(report_dir, 'training_history_comprehensive.png')
            if os.path.exists(img_path):
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title('Training History and Performance Analysis', fontsize=16, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 2: Data Analysis
            img_path = os.path.join(report_dir, 'data_analysis_comprehensive.png')
            if os.path.exists(img_path):
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title('Dataset Analysis and Feature Distribution', fontsize=16, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 3: Model Performance
            img_path = os.path.join(report_dir, 'model_performance_analysis.png')
            if os.path.exists(img_path):
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title('Model Performance Analysis', fontsize=16, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 4: Detailed Text Report
            fig = plt.figure(figsize=(8.5, 11))
            
            report_text = f"""
DETAILED TRAINING REPORT

=== EXECUTIVE SUMMARY ===
Model Version: {version}
Architecture: {params['architecture']}
Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Path: {model_path}

=== DATASET INFORMATION ===
Total Samples: {self.training_stats['data']['total_samples']:,}
Image Size: {params['img_size']}x{params['img_size']}
Batch Size: {params['batch_size']}

Age Statistics:
  - Range: {self.training_stats['data']['age_distribution']['min']:.0f} - {self.training_stats['data']['age_distribution']['max']:.0f} years
  - Mean: {self.training_stats['data']['age_distribution']['mean']:.1f} ± {self.training_stats['data']['age_distribution']['std']:.1f} years
  - Median: {self.training_stats['data']['age_distribution']['median']:.1f} years
  - Q1/Q3: {self.training_stats['data']['age_distribution']['q25']:.1f}/{self.training_stats['data']['age_distribution']['q75']:.1f} years

Gender Distribution:
  - Female: {self.training_stats['data']['gender_distribution']['female_count']:,} ({self.training_stats['data']['gender_distribution']['female_percentage']:.1f}%)
  - Male: {self.training_stats['data']['gender_distribution']['male_count']:,} ({self.training_stats['data']['gender_distribution']['male_percentage']:.1f}%)

=== MODEL ARCHITECTURE ===
Architecture: {self.training_stats['model']['architecture']}
Total Parameters: {self.training_stats['model']['total_params']:,}
Trainable Parameters: {self.training_stats['model']['trainable_params']:,}
Non-trainable Parameters: {self.training_stats['model']['non_trainable_params']:,}
Number of Layers: {self.training_stats['model']['layers']}
Input Shape: {self.training_stats['model']['input_shape']}

=== TRAINING CONFIGURATION ===
Initial Learning Rate: {params['learning_rate']}
Epochs Requested: {params['epochs']}
Epochs Completed: {self.training_stats['training']['epochs_completed']}
Best Epoch: {self.training_stats['training']['best_epoch']}
Training Duration: {self.training_stats['training']['duration']}

=== PERFORMANCE METRICS ===
Test Loss: {results['test_loss']:.4f}

Age Prediction:
  - Mean Absolute Error (MAE): {results['age_mae']:.2f} years
  - Root Mean Square Error (RMSE): {results['age_rmse']:.2f} years
  - R-squared (R²): {results['age_r2']:.3f}
  - Age-specific Loss: {results['age_loss']:.4f}

Gender Classification:
  - Accuracy: {results['gender_accuracy']:.1%}
  - Gender-specific Loss: {results['gender_loss']:.4f}

=== TRAINING INSIGHTS ===
Best Validation Loss: {self.training_stats['training']['best_val_loss']:.4f}
Final Training Loss: {self.training_stats['training']['final_train_loss']:.4f}
Final Validation Loss: {self.training_stats['training']['final_val_loss']:.4f}

Overfitting Analysis:
  - Gap between train/val loss: {abs(self.training_stats['training']['final_train_loss'] - self.training_stats['training']['final_val_loss']):.4f}
  - Early stopping triggered: {'Yes' if self.training_stats['training']['epochs_completed'] < params['epochs'] else 'No'}

=== FEATURE ENGINEERING INSIGHTS ===
Data Preprocessing Applied:
  - Image normalization (0-1 scaling)
  - Data augmentation during training
  - Train/Validation/Test split (70/15/15)

Age Distribution Balance:
  - Well-distributed across age groups
  - Potential bias towards younger ages (check age group chart)

Gender Balance:
  - {'Balanced' if abs(self.training_stats['data']['gender_distribution']['female_percentage'] - 50) < 10 else 'Imbalanced'}
  - May require class weighting if severely imbalanced

=== RECOMMENDATIONS ===
Based on the results:

1. Age Prediction Performance:
   {'Excellent' if results['age_mae'] < 5 else 'Good' if results['age_mae'] < 8 else 'Needs Improvement'}
   - MAE of {results['age_mae']:.1f} years is {'within acceptable range' if results['age_mae'] < 8 else 'higher than ideal'}

2. Gender Classification Performance:
   {'Excellent' if results['gender_accuracy'] > 0.9 else 'Good' if results['gender_accuracy'] > 0.8 else 'Needs Improvement'}
   - Accuracy of {results['gender_accuracy']:.1%} is {'satisfactory' if results['gender_accuracy'] > 0.8 else 'below expectations'}

3. Next Steps:
   - Consider data augmentation if overfitting observed
   - Experiment with different architectures for improvement
   - Collect more diverse data if performance is suboptimal
   - Fine-tune hyperparameters based on validation curves

=== MODEL DEPLOYMENT READINESS ===
Status: {'Ready for deployment' if results['age_mae'] < 8 and results['gender_accuracy'] > 0.8 else 'Requires further optimization'}

Confidence Level: {'High' if results['age_mae'] < 5 and results['gender_accuracy'] > 0.9 else 'Medium' if results['age_mae'] < 8 and results['gender_accuracy'] > 0.8 else 'Low'}
            """
            
            plt.text(0.05, 0.95, report_text, transform=fig.transFigure, fontsize=8, 
                    verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 5: Age Group Analysis
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('Detailed Age Group Analysis', fontsize=16, fontweight='bold')
            
            # Age group performance analysis
            age_groups = list(self.training_stats['data']['age_groups'].keys())
            age_counts = list(self.training_stats['data']['age_groups'].values())
            
            # Age group distribution
            axes[0, 0].bar(age_groups, age_counts, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Sample Count by Age Group', fontweight='bold')
            axes[0, 0].set_xlabel('Age Groups')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Age prediction error by age group
            age_pred = results['predictions']['age_pred']
            age_true = results['predictions']['age_true']
            errors_by_group = {}
            
            for i, age in enumerate(age_true):
                if age <= 10:
                    group = '0-10'
                elif age <= 20:
                    group = '11-20'
                elif age <= 30:
                    group = '21-30'
                elif age <= 40:
                    group = '31-40'
                elif age <= 50:
                    group = '41-50'
                elif age <= 60:
                    group = '51-60'
                elif age <= 70:
                    group = '61-70'
                else:
                    group = '71+'
                
                if group not in errors_by_group:
                    errors_by_group[group] = []
                errors_by_group[group].append(abs(age_pred[i] - age))
            
            group_mae = [np.mean(errors_by_group.get(group, [0])) for group in age_groups]
            
            axes[0, 1].bar(age_groups, group_mae, color='lightblue', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Age Prediction MAE by Age Group', fontweight='bold')
            axes[0, 1].set_xlabel('Age Groups')
            axes[0, 1].set_ylabel('MAE (years)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gender accuracy by age group
            gender_pred = results['predictions']['gender_pred']
            gender_true = results['predictions']['gender_true']
            gender_acc_by_group = {}
            
            for i, age in enumerate(age_true):
                if age <= 10:
                    group = '0-10'
                elif age <= 20:
                    group = '11-20'
                elif age <= 30:
                    group = '21-30'
                elif age <= 40:
                    group = '31-40'
                elif age <= 50:
                    group = '41-50'
                elif age <= 60:
                    group = '51-60'
                elif age <= 70:
                    group = '61-70'
                else:
                    group = '71+'
                
                if group not in gender_acc_by_group:
                    gender_acc_by_group[group] = []
                gender_acc_by_group[group].append(1 if gender_pred[i] == gender_true[i] else 0)
            
            group_gender_acc = [np.mean(gender_acc_by_group.get(group, [0])) for group in age_groups]
            
            axes[1, 0].bar(age_groups, group_gender_acc, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Gender Accuracy by Age Group', fontweight='bold')
            axes[1, 0].set_xlabel('Age Groups')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Summary statistics
            summary_text = f"""
AGE GROUP ANALYSIS SUMMARY

Sample Distribution:
Most represented: {age_groups[np.argmax(age_counts)]} ({max(age_counts)} samples)
Least represented: {age_groups[np.argmin(age_counts)]} ({min(age_counts)} samples)

Age Prediction Performance:
Best performing group: {age_groups[np.argmin(group_mae)]} (MAE: {min(group_mae):.1f} years)
Worst performing group: {age_groups[np.argmax(group_mae)]} (MAE: {max(group_mae):.1f} years)

Gender Classification Performance:
Best performing group: {age_groups[np.argmax(group_gender_acc)]} (Acc: {max(group_gender_acc):.1%})
Worst performing group: {age_groups[np.argmin(group_gender_acc)]} (Acc: {min(group_gender_acc):.1%})

Data Quality Insights:
- Age groups with <100 samples may have unreliable performance metrics
- Consider collecting more data for underrepresented groups
- Performance variations across age groups indicate potential model bias
            """
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Comprehensive PDF report generated: {pdf_path}")
        return pdf_path
    
    def generate_report(self, params, results, model_path, version):
        """Generate comprehensive training report with plots and PDF"""
        logger.info("Generating comprehensive training report...")
        
        # Create reports directory
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(reports_dir, f"training_report_v{version}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate comprehensive plots
        self.create_comprehensive_plots(results, report_dir)
        
        # Generate detailed text summary
        self.generate_detailed_text_report(params, results, model_path, version, report_dir)
        
        # Generate PDF report
        pdf_path = self.generate_pdf_report(params, results, model_path, version, report_dir)
        
        # Generate CSV data export
        self.export_data_to_csv(results, report_dir)
        
        logger.info(f"Comprehensive training report generated in: {report_dir}")
        return report_dir, pdf_path
    
    def generate_detailed_text_report(self, params, results, model_path, version, report_dir):
        """Generate detailed text report"""
        report_path = os.path.join(report_dir, 'detailed_report_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AGE AND GENDER PREDICTION MODEL - COMPREHENSIVE TRAINING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Version: {version}\n")
            f.write(f"Model Path: {model_path}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"The {params['architecture']} model was trained for age and gender prediction ")
            f.write(f"achieving an age MAE of {results['age_mae']:.2f} years and gender accuracy of ")
            f.write(f"{results['gender_accuracy']:.1%}. Training completed in {self.training_stats['training']['duration']} ")
            f.write(f"over {self.training_stats['training']['epochs_completed']} epochs.\n\n")
            
            # Dataset Analysis
            f.write("DATASET ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Samples: {self.training_stats['data']['total_samples']:,}\n")
            f.write(f"Image Resolution: {params['img_size']}x{params['img_size']} pixels\n")
            f.write(f"Batch Size: {params['batch_size']}\n\n")
            
            f.write("Age Statistics:\n")
            age_stats = self.training_stats['data']['age_distribution']
            f.write(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years\n")
            f.write(f"  Mean: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years\n")
            f.write(f"  Median: {age_stats['median']:.1f} years\n")
            f.write(f"  Quartiles: Q1={age_stats['q25']:.1f}, Q3={age_stats['q75']:.1f}\n\n")
            
            f.write("Gender Distribution:\n")
            gender_stats = self.training_stats['data']['gender_distribution']
            f.write(f"  Female: {gender_stats['female_count']:,} ({gender_stats['female_percentage']:.1f}%)\n")
            f.write(f"  Male: {gender_stats['male_count']:,} ({gender_stats['male_percentage']:.1f}%)\n\n")
            
            f.write("Age Group Distribution:\n")
            for group, count in self.training_stats['data']['age_groups'].items():
                percentage = (count / self.training_stats['data']['total_samples']) * 100
                f.write(f"  {group}: {count:,} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Model Architecture
            f.write("MODEL ARCHITECTURE\n")
            f.write("-" * 40 + "\n")
            model_info = self.training_stats['model']
            f.write(f"Architecture: {model_info['architecture']}\n")
            f.write(f"Total Parameters: {model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
            f.write(f"Non-trainable Parameters: {model_info['non_trainable_params']:,}\n")
            f.write(f"Number of Layers: {model_info['layers']}\n")
            f.write(f"Input Shape: {model_info['input_shape']}\n")
            f.write(f"Initial Learning Rate: {model_info['learning_rate']}\n\n")
            
            # Training Process
            f.write("TRAINING PROCESS\n")
            f.write("-" * 40 + "\n")
            training_info = self.training_stats['training']
            f.write(f"Start Time: {training_info['start_time']}\n")
            f.write(f"End Time: {training_info['end_time']}\n")
            f.write(f"Duration: {training_info['duration']}\n")
            f.write(f"Epochs Requested: {params['epochs']}\n")
            f.write(f"Epochs Completed: {training_info['epochs_completed']}\n")
            f.write(f"Best Epoch: {training_info['best_epoch']}\n")
            f.write(f"Early Stopping: {'Yes' if training_info['epochs_completed'] < params['epochs'] else 'No'}\n\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Test Loss: {results['test_loss']:.4f}\n\n")
            
            f.write("Age Prediction Performance:\n")
            f.write(f"  Mean Absolute Error (MAE): {results['age_mae']:.2f} years\n")
            f.write(f"  Root Mean Square Error (RMSE): {results['age_rmse']:.2f} years\n")
            f.write(f"  R-squared (R²): {results['age_r2']:.3f}\n")
            f.write(f"  Age-specific Loss: {results['age_loss']:.4f}\n\n")
            
            f.write("Gender Classification Performance:\n")
            f.write(f"  Accuracy: {results['gender_accuracy']:.1%}\n")
            f.write(f"  Gender-specific Loss: {results['gender_loss']:.4f}\n\n")
            
            # Confusion Matrix Details
            cm = results['gender_confusion_matrix']
            f.write("Gender Confusion Matrix:\n")
            f.write("                Predicted\n")
            f.write("              Female  Male\n")
            f.write(f"True Female   {cm[0][0]:6d}  {cm[0][1]:4d}\n")
            f.write(f"     Male     {cm[1][0]:6d}  {cm[1][1]:4d}\n\n")
            
            # Training Insights
            f.write("TRAINING INSIGHTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Validation Loss: {training_info['best_val_loss']:.4f} (Epoch {training_info['best_epoch']})\n")
            f.write(f"Final Training Loss: {training_info['final_train_loss']:.4f}\n")
            f.write(f"Final Validation Loss: {training_info['final_val_loss']:.4f}\n")
            
            loss_gap = abs(training_info['final_train_loss'] - training_info['final_val_loss'])
            f.write(f"Train/Val Loss Gap: {loss_gap:.4f}\n")
            
            if loss_gap > 0.1:
                f.write("  ⚠️  Large gap suggests potential overfitting\n")
            elif loss_gap < 0.02:
                f.write("  ✅ Small gap indicates good generalization\n")
            else:
                f.write("  ✅ Moderate gap is acceptable\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS AND INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Age prediction analysis
            if results['age_mae'] < 5:
                f.write("✅ Age Prediction: Excellent performance (MAE < 5 years)\n")
            elif results['age_mae'] < 8:
                f.write("✅ Age Prediction: Good performance (MAE < 8 years)\n")
            else:
                f.write("⚠️  Age Prediction: Room for improvement (MAE > 8 years)\n")
            
            # Gender classification analysis
            if results['gender_accuracy'] > 0.9:
                f.write("✅ Gender Classification: Excellent performance (>90% accuracy)\n")
            elif results['gender_accuracy'] > 0.8:
                f.write("✅ Gender Classification: Good performance (>80% accuracy)\n")
            else:
                f.write("⚠️  Gender Classification: Needs improvement (<80% accuracy)\n")
            
            f.write("\nNext Steps:\n")
            f.write("1. Monitor for overfitting if train/val gap is large\n")
            f.write("2. Consider data augmentation for better generalization\n")
            f.write("3. Experiment with different architectures if performance is suboptimal\n")
            f.write("4. Collect more diverse data for underrepresented age groups\n")
            f.write("5. Fine-tune hyperparameters based on validation curves\n")
            
            # Deployment readiness
            f.write(f"\nDeployment Readiness: ")
            if results['age_mae'] < 8 and results['gender_accuracy'] > 0.8:
                f.write("✅ Ready for deployment\n")
            else:
                f.write("⚠️  Requires further optimization\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
    
    def export_data_to_csv(self, results, report_dir):
        """Export training data and results to CSV files"""
        
        # Training history CSV
        if self.history:
            history_df = pd.DataFrame(self.history.history)
            history_df['epoch'] = range(1, len(history_df) + 1)
            history_df.to_csv(os.path.join(report_dir, 'training_history.csv'), index=False)
        
        # Predictions CSV
        predictions_df = pd.DataFrame({
            'age_true': results['predictions']['age_true'],
            'age_predicted': results['predictions']['age_pred'],
            'age_error': results['predictions']['age_pred'] - results['predictions']['age_true'],
            'age_abs_error': np.abs(results['predictions']['age_pred'] - results['predictions']['age_true']),
            'gender_true': results['predictions']['gender_true'],
            'gender_predicted': results['predictions']['gender_pred'],
            'gender_correct': (results['predictions']['gender_pred'] == results['predictions']['gender_true']).astype(int)
        })
        predictions_df.to_csv(os.path.join(report_dir, 'test_predictions.csv'), index=False)
        
        # Model summary CSV
        model_summary_df = pd.DataFrame([{
            'metric': 'age_mae',
            'value': results['age_mae'],
            'unit': 'years'
        }, {
            'metric': 'age_rmse',
            'value': results['age_rmse'],
            'unit': 'years'
        }, {
            'metric': 'age_r2',
            'value': results['age_r2'],
            'unit': 'correlation'
        }, {
            'metric': 'gender_accuracy',
            'value': results['gender_accuracy'],
            'unit': 'percentage'
        }])
        model_summary_df.to_csv(os.path.join(report_dir, 'model_metrics.csv'), index=False)
        
        logger.info("Data exported to CSV files")
    
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