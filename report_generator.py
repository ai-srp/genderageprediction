#!/usr/bin/env python3
"""
Report generation utilities for Age and Gender Prediction Model
"""
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

def _create_fallback_architecture_image(report_dir):
    """Create a simple fallback image when the detailed neural network diagram fails"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Create a simple figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Add text explaining the situation
    ax.text(0.5, 0.6, "Neural Network Architecture", 
            fontsize=16, fontweight='bold', ha='center', transform=fig.transFigure)
    ax.text(0.5, 0.5, "Detailed visualization unavailable", 
            fontsize=14, ha='center', transform=fig.transFigure)
    ax.text(0.5, 0.4, "The model structure might be complex or unusual.", 
            fontsize=12, ha='center', transform=fig.transFigure)
    
    # Save the basic diagram
    arch_img_path = os.path.join(report_dir, 'nn_architecture.png')
    plt.savefig(arch_img_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return arch_img_path

def plot_model_architecture(trainer, report_dir):
    """Plot a detailed neural network architecture diagram and save as image."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import logging
    import sys
    
    logger = logging.getLogger(__name__)
    
    try:
        model = trainer.model
        logger.info("Starting neural network architecture diagram generation")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        # Define colors for different layer types
        color_map = {
            'Conv2D': '#3498db',         # Blue
            'MaxPooling2D': '#2ecc71',   # Green
            'Dense': '#e74c3c',          # Red
            'Dropout': '#95a5a6',        # Gray
            'Flatten': '#9b59b6',        # Purple
            'BatchNormalization': '#f39c12',  # Orange
            'InputLayer': '#1abc9c',     # Turquoise
            'Activation': '#34495e',     # Dark Blue
            'Add': '#f1c40f',           # Yellow
            'Concatenate': '#e67e22',    # Dark Orange
        }
        
        # Define layout parameters
        num_layers = len(model.layers)
        x_start = 0.1
        x_end = 0.9
        y_mid = 0.5
        layer_spacing = (x_end - x_start) / max(1, num_layers - 1) if num_layers > 1 else 0.5
        
        # Simple fixed height for all boxes to avoid issues
        box_width = 0.06
        box_height = 0.15
        
        # Draw layers
        layer_positions = []
        
        for i, layer in enumerate(model.layers):
            try:
                # Get basic info that should be safe for any layer
                layer_type = layer.__class__.__name__
                params_count = layer.count_params()
                
                # Calculate position
                x = x_start + i * layer_spacing
                y = y_mid
                
                # Get color for this layer type
                color = color_map.get(layer_type, '#cccccc')
                
                # Draw layer box
                rect = mpatches.FancyBboxPatch(
                    (x - box_width/2, y - box_height/2), 
                    box_width, box_height, 
                    boxstyle="round,pad=0.02", 
                    ec="black", fc=color, alpha=0.8
                )
                ax.add_patch(rect)
                
                # Store position for connections
                layer_positions.append((x, y, box_width, box_height))
                
                # Add layer type name
                ax.text(x, y + box_height/2 + 0.02, layer_type, 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                  # Safely try to add shape information
                try:
                    # Convert any output shape object to string safely
                    shape_text = "Shape: "
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                        if output_shape is None:
                            shape_text += "unknown"
                        elif isinstance(output_shape, (list, tuple)):
                            # It's already an iterable, convert to string
                            shape_text += str(tuple(output_shape)).replace('None', 'batch')
                        elif isinstance(output_shape, int):
                            # It's a single integer, not subscriptable
                            shape_text += f"({output_shape},)"
                        elif hasattr(output_shape, 'shape'):
                            # Try to get shape attribute if present
                            shape_text += str(output_shape.shape).replace('None', 'batch')
                        else:
                            # Default case: just stringify it
                            shape_text += str(output_shape)
                    else:
                        shape_text += "unknown"
                        
                    ax.text(x, y, shape_text, ha='center', va='center', 
                            fontsize=8, color='black')
                except Exception as e:
                    # Log the specific error for debugging
                    logger.debug(f"Error adding shape info for layer {layer.__class__.__name__}: {str(e)}")
                    # If shape info fails, just skip it
                    pass
                
                # Add parameter count
                if params_count > 0:
                    params_text = f"Params: {params_count:,}" if params_count < 10000 else f"Params: {params_count/1000:.1f}K"
                    ax.text(x, y - box_height/2 - 0.02, params_text, 
                            ha='center', va='top', fontsize=8)
            
            except Exception as e:
                # If a specific layer fails, log it but continue with other layers
                logger.warning(f"Error processing layer {i}: {str(e)}")
                continue
          # Draw connections between layers
        for i in range(len(layer_positions) - 1):
            try:
                x1, y1, w1, h1 = layer_positions[i]
                x2, y2, w2, h2 = layer_positions[i + 1]
                
                # Draw simple line connecting layers
                ax.plot([x1 + w1/2, x2 - w2/2], [y1, y2], 'k-', alpha=0.6)
            except Exception as e:
                # Log the specific error but continue with other connections
                logger.debug(f"Error drawing connection between layers {i} and {i+1}: {str(e)}")
                continue
        
        # Add title and overall info
        plt.title('Neural Network Architecture', fontsize=14, fontweight='bold')
        
        # Add model summary stats
        try:
            model_info = trainer.training_stats.get('model', {})
            total_params = model_info.get('total_params', model.count_params())
            trainable_params = model_info.get('trainable_params', 0)
            
            summary_text = f"Total parameters: {total_params:,}"
            if trainable_params > 0:
                summary_text += f" | Trainable: {trainable_params:,}"
                
            plt.figtext(0.5, 0.95, summary_text, ha='center', fontsize=10)
        except:
            # If adding summary fails, just skip it
            pass
            
        # Save diagram
        arch_img_path = os.path.join(report_dir, 'nn_architecture.png')
        plt.savefig(arch_img_path, bbox_inches='tight', dpi=200)
        plt.close()
        
        logger.info("Neural network architecture diagram generated successfully")
        return arch_img_path
        
    except Exception as e:
        # If anything fails, return a fallback diagram
        logger.error(f"Error generating neural network diagram: {str(e)}")
        return _create_fallback_architecture_image(report_dir)

def create_comprehensive_plots(trainer, results, report_dir):
    """Create comprehensive visualization plots"""
    # 1. Training History with Multiple Metrics
    if trainer.history:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training History and Performance Analysis', fontsize=16, fontweight='bold')
        
        # Total loss
        axes[0, 0].plot(trainer.history.history['loss'], label='Train')
        axes[0, 0].plot(trainer.history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Age loss
        if 'age_loss' in trainer.history.history:
            axes[0, 1].plot(trainer.history.history['age_loss'], label='Train')
            axes[0, 1].plot(trainer.history.history['val_age_loss'], label='Validation')
            axes[0, 1].set_title('Age Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Gender loss
        if 'gender_loss' in trainer.history.history:
            axes[0, 2].plot(trainer.history.history['gender_loss'], label='Train')
            axes[0, 2].plot(trainer.history.history['val_gender_loss'], label='Validation')
            axes[0, 2].set_title('Gender Loss', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Age MAE
        if 'age_mae' in trainer.history.history:
            axes[1, 0].plot(trainer.history.history['age_mae'], label='Train')
            axes[1, 0].plot(trainer.history.history['val_age_mae'], label='Validation')
            axes[1, 0].set_title('Age Mean Absolute Error', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE (years)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gender accuracy
        if 'gender_accuracy' in trainer.history.history:
            axes[1, 1].plot(trainer.history.history['gender_accuracy'], label='Train')
            axes[1, 1].plot(trainer.history.history['val_gender_accuracy'], label='Validation')
            axes[1, 1].set_title('Gender Accuracy', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in trainer.history.history:
            axes[1, 2].plot(trainer.history.history['lr'])
            axes[1, 2].set_title('Learning Rate', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'training_history_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Data Analysis Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Analysis and Feature Distribution', fontsize=16, fontweight='bold')
    
    # Age distribution (histogram)
    axes[0, 0].hist(trainer.training_stats['data']['ages'], bins=20, alpha=0.7, 
                  color='skyblue', edgecolor='black')
    axes[0, 0].axvline(trainer.training_stats['data']['age_distribution']['mean'], 
                      color='red', linestyle='--', label=f"Mean: {trainer.training_stats['data']['age_distribution']['mean']:.1f}")
    axes[0, 0].set_title('Age Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gender distribution (pie chart)
    gender_counts = [trainer.training_stats['data']['gender_distribution']['female_count'],
                    trainer.training_stats['data']['gender_distribution']['male_count']]
    axes[0, 1].pie(gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%', 
                  colors=['pink', 'lightblue'], startangle=90)
    axes[0, 1].set_title('Gender Distribution', fontweight='bold')
    
    # Age groups bar chart
    age_groups = trainer.training_stats['data']['age_groups']
    axes[0, 2].bar(age_groups.keys(), age_groups.values(), color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Age Groups Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('Age Groups')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Age vs Gender scatter
    male_ages = [age for i, age in enumerate(trainer.training_stats['data']['ages']) 
                if trainer.training_stats['data']['genders'][i] == 1]
    female_ages = [age for i, age in enumerate(trainer.training_stats['data']['ages']) 
                  if trainer.training_stats['data']['genders'][i] == 0]
    
    axes[1, 0].hist([female_ages, male_ages], bins=20, alpha=0.7, 
                   label=['Female', 'Male'], color=['pink', 'lightblue'], edgecolor='black')
    axes[1, 0].set_title('Age Distribution by Gender', fontweight='bold')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot for age by gender
    gender_labels = ['Female' if g == 0 else 'Male' for g in trainer.training_stats['data']['genders']]
    age_gender_df = pd.DataFrame({'Age': trainer.training_stats['data']['ages'], 'Gender': gender_labels})
    
    sns.boxplot(data=age_gender_df, x='Gender', y='Age', ax=axes[1, 1])
    axes[1, 1].set_title('Age Distribution Box Plot by Gender', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Data quality indicators
    axes[1, 2].text(0.1, 0.9, f"Total Samples: {trainer.training_stats['data']['total_samples']}", 
                   transform=axes[1, 2].transAxes, fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.8, f"Age Range: {trainer.training_stats['data']['age_distribution']['min']:.0f} - {trainer.training_stats['data']['age_distribution']['max']:.0f} years", 
                   transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.7, f"Age Mean ± Std: {trainer.training_stats['data']['age_distribution']['mean']:.1f} ± {trainer.training_stats['data']['age_distribution']['std']:.1f}", 
                   transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.6, f"Gender Balance: {trainer.training_stats['data']['gender_distribution']['female_percentage']:.1f}% F / {trainer.training_stats['data']['gender_distribution']['male_percentage']:.1f}% M", 
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
    model_info = trainer.training_stats['model']
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
    axes[1, 1].text(0.1, 0.3, f"Training Duration: {trainer.training_stats['training']['duration']}", 
                   transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Model Configuration', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'model_performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Comprehensive plots generated successfully")

def generate_pdf_report(trainer, params, results, model_path, version, report_dir):
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
        
        plt.text(0.5, 0.15, f"Training Duration: {trainer.training_stats['training']['duration']}", ha='center', fontsize=12, transform=fig.transFigure)
        plt.text(0.5, 0.1, f"Total Parameters: {trainer.training_stats['model']['total_params']:,}", ha='center', fontsize=12, transform=fig.transFigure)
        
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
            plt.close()        # Insert neural network architecture diagram (with error handling)
        try:
            logger.info("Adding neural network architecture diagram to PDF")
            arch_img_path = os.path.join(report_dir, 'nn_architecture.png')
            
            # Generate the architecture diagram if it doesn't exist
            # TODO: Commenting out the line below to avoid generating Network Architecture image again
            # if not os.path.exists(arch_img_path):
            #     arch_img_path = plot_model_architecture(trainer, report_dir)
            
            # Add to PDF if available
            if os.path.exists(arch_img_path):
                fig = plt.figure(figsize=(11, 6))
                img = plt.imread(arch_img_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title('Neural Network Architecture (Detailed View)', fontsize=14, fontweight='bold')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                logger.info("Neural network architecture diagram added to PDF")
            else:
                logger.warning("No neural network architecture diagram available to add to PDF")
        except Exception as e:
            logger.error(f"Error adding neural network diagram to PDF: {str(e)}")
            # Continue with the report even if this fails
          # Additional Keras visualization (if available)
        try:
            # Try to create Keras visualization if it doesn't exist
            keras_vis_path = os.path.join(report_dir, 'keras_model_visualization.png')
            if not os.path.exists(keras_vis_path):
                keras_vis_path = create_keras_model_visualization(trainer, report_dir)
                
            # Add to PDF if available
            if keras_vis_path and os.path.exists(keras_vis_path):
                fig = plt.figure(figsize=(11, 8))
                img = plt.imread(keras_vis_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title('Neural Network Architecture (Layer Connections)', fontsize=14, fontweight='bold')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                logger.info("Keras visualization added to PDF report")
        except Exception as e:
            logger.error(f"Error adding Keras visualization to PDF: {str(e)}")
            # Continue with the report even if this fails
        
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
Total Samples: {trainer.training_stats['data']['total_samples']:,}
Image Size: {params['img_size']}x{params['img_size']}
Batch Size: {params['batch_size']}

Age Statistics:
  - Range: {trainer.training_stats['data']['age_distribution']['min']:.0f} - {trainer.training_stats['data']['age_distribution']['max']:.0f} years
  - Mean: {trainer.training_stats['data']['age_distribution']['mean']:.1f} ± {trainer.training_stats['data']['age_distribution']['std']:.1f} years
  - Median: {trainer.training_stats['data']['age_distribution']['median']:.1f} years
  - Q1/Q3: {trainer.training_stats['data']['age_distribution']['q25']:.1f}/{trainer.training_stats['data']['age_distribution']['q75']:.1f} years

Gender Distribution:
  - Female: {trainer.training_stats['data']['gender_distribution']['female_count']:,} ({trainer.training_stats['data']['gender_distribution']['female_percentage']:.1f}%)
  - Male: {trainer.training_stats['data']['gender_distribution']['male_count']:,} ({trainer.training_stats['data']['gender_distribution']['male_percentage']:.1f}%)

=== MODEL ARCHITECTURE ===
Architecture: {trainer.training_stats['model']['architecture']}
Total Parameters: {trainer.training_stats['model']['total_params']:,}
Trainable Parameters: {trainer.training_stats['model']['trainable_params']:,}
Non-trainable Parameters: {trainer.training_stats['model']['non_trainable_params']:,}
Number of Layers: {trainer.training_stats['model']['layers']}
Input Shape: {trainer.training_stats['model']['input_shape']}

=== TRAINING CONFIGURATION ===
Initial Learning Rate: {params['learning_rate']}
Epochs Requested: {params['epochs']}
Epochs Completed: {trainer.training_stats['training']['epochs_completed']}
Best Epoch: {trainer.training_stats['training']['best_epoch']}
Training Duration: {trainer.training_stats['training']['duration']}

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
Best Validation Loss: {trainer.training_stats['training']['best_val_loss']:.4f}
Final Training Loss: {trainer.training_stats['training']['final_train_loss']:.4f}
Final Validation Loss: {trainer.training_stats['training']['final_val_loss']:.4f}

Overfitting Analysis:
  - Gap between train/val loss: {abs(trainer.training_stats['training']['final_train_loss'] - trainer.training_stats['training']['final_val_loss']):.4f}
  - Early stopping triggered: {'Yes' if trainer.training_stats['training']['epochs_completed'] < params['epochs'] else 'No'}

=== FEATURE ENGINEERING INSIGHTS ===
Data Preprocessing Applied:
  - Image normalization (0-1 scaling)
  - Data augmentation during training
  - Train/Validation/Test split (72/8/20)

Age Distribution Balance:
  - Well-distributed across age groups
  - Potential bias towards younger ages (check age group chart)

Gender Balance:
  - {'Balanced' if abs(trainer.training_stats['data']['gender_distribution']['female_percentage'] - 50) < 10 else 'Imbalanced'}
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
        age_groups = list(trainer.training_stats['data']['age_groups'].keys())
        age_counts = list(trainer.training_stats['data']['age_groups'].values())
        
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

def generate_detailed_text_report(trainer, params, results, model_path, version, report_dir):
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
        f.write(f"{results['gender_accuracy']:.1%}. Training completed in {trainer.training_stats['training']['duration']} ")
        f.write(f"over {trainer.training_stats['training']['epochs_completed']} epochs.\n\n")
        
        # Dataset Analysis
        f.write("DATASET ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {trainer.training_stats['data']['total_samples']:,}\n")
        f.write(f"Image Resolution: {params['img_size']}x{params['img_size']} pixels\n")
        f.write(f"Batch Size: {params['batch_size']}\n\n")
        
        f.write("Age Statistics:\n")
        age_stats = trainer.training_stats['data']['age_distribution']
        f.write(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years\n")
        f.write(f"  Mean: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years\n")
        f.write(f"  Median: {age_stats['median']:.1f} years\n")
        f.write(f"  Quartiles: Q1={age_stats['q25']:.1f}, Q3={age_stats['q75']:.1f}\n\n")
        
        f.write("Gender Distribution:\n")
        gender_stats = trainer.training_stats['data']['gender_distribution']
        f.write(f"  Female: {gender_stats['female_count']:,} ({gender_stats['female_percentage']:.1f}%)\n")
        f.write(f"  Male: {gender_stats['male_count']:,} ({gender_stats['male_percentage']:.1f}%)\n\n")
        
        f.write("Age Group Distribution:\n")
        for group, count in trainer.training_stats['data']['age_groups'].items():
            percentage = (count / trainer.training_stats['data']['total_samples']) * 100
            f.write(f"  {group}: {count:,} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Model Architecture
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 40 + "\n")
        model_info = trainer.training_stats['model']
        f.write(f"Architecture: {model_info['architecture']}\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
        f.write(f"Non-trainable Parameters: {model_info['non_trainable_params']:,}\n")
        f.write(f"Number of Layers: {model_info['layers']}\n")
        f.write(f"Input Shape: {model_info['input_shape']}\n")
        f.write(f"Initial Learning Rate: {model_info['learning_rate']}\n\n")
          # Detailed model architecture information
        model = trainer.model
        f.write("DETAILED MODEL ARCHITECTURE\n")
        f.write("-" * 40 + "\n")
        
        f.write("Layer Sequence:\n")
        layer_index = 0
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            output_shape = layer.output_shape
            layer_shape = str(output_shape).replace('(None, ', '(batch_size, ')
            params = layer.count_params()
            
            # Format layer info with proper indentation
            f.write(f"  {layer_index:2d}. {layer_type:<15} | Shape: {layer_shape:<25} | Params: {params:,}\n")
            
            # Add activation function info if available
            if hasattr(layer, 'activation') and layer.activation is not None:
                act = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                if act != 'linear':
                    f.write(f"      ↳ Activation: {act}\n")
            
            # Add kernel/filter info for Conv layers
            if 'Conv' in layer_type and hasattr(layer, 'kernel_size'):
                f.write(f"      ↳ Kernel size: {layer.kernel_size}, Filters: {layer.filters}\n")
                if hasattr(layer, 'strides') and layer.strides != (1, 1):
                    f.write(f"      ↳ Strides: {layer.strides}\n")
            
            # Add pool size info for pooling layers
            if 'Pooling' in layer_type and hasattr(layer, 'pool_size'):
                f.write(f"      ↳ Pool size: {layer.pool_size}\n")
                if hasattr(layer, 'strides') and layer.strides != layer.pool_size:
                    f.write(f"      ↳ Strides: {layer.strides}\n")
            
            # Add rate info for dropout layers
            if layer_type == 'Dropout' and hasattr(layer, 'rate'):
                f.write(f"      ↳ Dropout rate: {layer.rate}\n")
            
            # Add units info for dense layers
            if layer_type == 'Dense' and hasattr(layer, 'units'):
                f.write(f"      ↳ Units: {layer.units}\n")
            
            layer_index += 1
        f.write("\n")
        
        # Training Process
        f.write("TRAINING PROCESS\n")
        f.write("-" * 40 + "\n")
        training_info = trainer.training_stats['training']
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
    
    logger.info(f"Detailed text report generated: {report_path}")
    return report_path

def create_keras_model_visualization(trainer, report_dir):
    """
    Create a visualization of the model using Keras' built-in utils.plot_model or
    a simplified visualization if dependencies aren't available.
    """
    import os
    import logging
    import importlib.util
    import matplotlib.pyplot as plt
    
    logger = logging.getLogger(__name__)
    vis_path = os.path.join(report_dir, 'keras_model_visualization.png')
    
    # Check if TensorFlow is available
    if importlib.util.find_spec("tensorflow") is None:
        logger.warning("TensorFlow not found, creating simplified model visualization")
        return create_simplified_model_visualization(trainer, report_dir)
    
    # Check for Pydot/Graphviz
    has_pydot = importlib.util.find_spec("pydot") is not None
    if not has_pydot:
        logger.warning("pydot package not found, creating simplified model visualization")
        return create_simplified_model_visualization(trainer, report_dir)
        
    try:
        # Import and use tensorflow
        import tensorflow as tf
        
        # Get plot_model from appropriate path
        try:
            if hasattr(tf, 'keras'):
                # TF 2.x
                plot_model = tf.keras.utils.plot_model
            else:
                # Try standalone Keras
                import keras
                plot_model = keras.utils.plot_model
                
            model = trainer.model
            
            # Use Keras' built-in visualization function
            plot_model(
                model, 
                to_file=vis_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',  # Top to Bottom layout
                expand_nested=True,
                dpi=96
            )
            
            # Check if file was created successfully
            if os.path.exists(vis_path) and os.path.getsize(vis_path) > 0:
                logger.info(f"Keras model visualization created at {vis_path}")
                return vis_path
            else:
                logger.warning("Keras plot_model ran but did not create a valid image")
                return create_simplified_model_visualization(trainer, report_dir)
                
        except Exception as inner_e:
            logger.warning(f"Error using Keras plot_model: {str(inner_e)}")
            return create_simplified_model_visualization(trainer, report_dir)
            
    except Exception as e:
        logger.warning(f"Could not create Keras model visualization: {str(e)}")
        return create_simplified_model_visualization(trainer, report_dir)


def create_simplified_model_visualization(trainer, report_dir):
    """Create a simplified model visualization as fallback when Keras tools are not available."""
    import os
    import logging
    import matplotlib.pyplot as plt
    
    logger = logging.getLogger(__name__)
    
    try:
        model = trainer.model
        vis_path = os.path.join(report_dir, 'keras_model_visualization.png')
        
        # Create a figure to visualize layers as a simple block diagram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get layer information
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        num_layers = len(layer_names)
        
        # Visual attributes
        box_height = 0.5
        box_width = 2.5
        vertical_spacing = 0.7
        
        # Color mapping for different layer types
        color_map = {
            'Conv2D': '#3498db',         # Blue
            'MaxPooling2D': '#2ecc71',   # Green
            'Dense': '#e74c3c',          # Red
            'Dropout': '#95a5a6',        # Gray
            'Flatten': '#9b59b6',        # Purple
            'BatchNormalization': '#f39c12',  # Orange
            'InputLayer': '#1abc9c',     # Turquoise
            'Activation': '#34495e',     # Dark Blue
        }
        
        # Default color for other layer types
        default_color = '#cccccc'
        
        # Draw each layer as a box
        for i, layer_name in enumerate(layer_names):
            y_position = num_layers - i
            
            # Get layer color
            color = color_map.get(layer_name, default_color)
            
            # Draw box for the layer
            rect = plt.Rectangle((1, y_position), box_width, box_height, 
                               facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Add layer name
            plt.text(1 + box_width/2, y_position + box_height/2, layer_name,
                    ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Add additional info if available
            try:
                layer = model.layers[i]
                params = layer.count_params()
                info_text = f"Params: {params:,}"
                plt.text(1 + box_width/2, y_position + 0.1, info_text,
                        ha='center', va='center', fontsize=7)
            except:
                pass
            
            # Draw arrow to next layer
            if i < num_layers - 1:
                ax.arrow(1 + box_width/2, y_position, 0, -vertical_spacing,
                        head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Set plot limits
        ax.set_xlim(0.5, box_width + 1.5)
        ax.set_ylim(0.5, num_layers + 1.5)
        
        # Add title
        plt.title('Model Architecture Overview', fontsize=14, fontweight='bold')
        
        # Remove axes
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created simplified model visualization at {vis_path}")
        return vis_path
    except Exception as e:
        logger.warning(f"Could not create simplified model visualization: {str(e)}")
        return None

def export_data_to_csv(trainer, results, report_dir):
    """Export training data and results to CSV files"""
    
    # Training history CSV
    if trainer.history:
        history_df = pd.DataFrame(trainer.history.history)
        history_df['epoch'] = range(1, len(history_df) + 1)
        history_df.to_csv(os.path.join(report_dir, 'training_history.csv'), index=False, encoding='utf-8')
    
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
    predictions_df.to_csv(os.path.join(report_dir, 'test_predictions.csv'), index=False, encoding='utf-8')
    
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
    model_summary_df.to_csv(os.path.join(report_dir, 'model_metrics.csv'), index=False, encoding='utf-8')
    
    logger.info("Data exported to CSV files")

def generate_report(trainer, params, results, model_path, version):
    """Generate comprehensive training report with plots and PDF"""
    logger.info("Generating comprehensive training report...")
    
    # Create reports directory
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(reports_dir, f"training_report_v{version}_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate each component with proper error handling
    components_generated = {
        "plots": False,
        "text_report": False,
        "pdf_report": False,
        "csv_export": False
    }
    
    # Create comprehensive plots
    try:
        logger.info("Generating comprehensive plots...")
        create_comprehensive_plots(trainer, results, report_dir)
        components_generated["plots"] = True
        logger.info("Plots generated successfully")
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        # Continue with report generation
    
    # Generate detailed text summary
    try:
        logger.info("Generating detailed text report...")
        generate_detailed_text_report(trainer, params, results, model_path, version, report_dir)
        components_generated["text_report"] = True
        logger.info("Text report generated successfully")
    except Exception as e:
        logger.error(f"Error generating text report: {str(e)}")
        # Continue with report generation
    
    # Generate PDF report
    try:
        logger.info("Generating PDF report...")
        pdf_path = generate_pdf_report(trainer, params, results, model_path, version, report_dir)
        components_generated["pdf_report"] = True
        logger.info("PDF report generated successfully")
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        pdf_path = None
        # Continue with report generation
    
    # Generate CSV data export
    try:
        logger.info("Exporting data to CSV...")
        export_data_to_csv(trainer, results, report_dir)
        components_generated["csv_export"] = True
        logger.info("Data exported successfully")
    except Exception as e:
        logger.error(f"Error exporting data to CSV: {str(e)}")
        # Continue with report generation
    
    # Log summary of report generation
    success_count = sum(components_generated.values())
    logger.info(f"Report generation complete: {success_count}/4 components successful")
    for component, success in components_generated.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  - {component}: {status}")
    
    logger.info(f"Comprehensive training report generated in: {report_dir}")
    return report_dir, pdf_path
