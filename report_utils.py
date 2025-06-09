import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

def plot_model_architecture(trainer, report_dir):
    """Plot a simple neural network architecture diagram and save as image."""
    import matplotlib.patches as mpatches
    model = trainer.model
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    y = 0.5
    x = 0.1
    layer_gap = 0.12
    for i, layer in enumerate(model.layers):
        color = 'skyblue' if 'Conv' in layer.__class__.__name__ else 'lightgreen' if 'Dense' in layer.__class__.__name__ else 'lightgray'
        rect = mpatches.FancyBboxPatch((x, y), 0.08, 0.2, boxstyle="round,pad=0.02", ec="k", fc=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+0.04, y+0.1, layer.__class__.__name__, ha='center', va='center', fontsize=8)
        ax.text(x+0.04, y+0.03, f"{layer.output_shape[-1] if hasattr(layer, 'output_shape') else ''}", ha='center', va='center', fontsize=7, color='gray')
        x += layer_gap
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Neural Network Architecture (Layer Types)', fontsize=12)
    arch_img_path = os.path.join(report_dir, 'nn_architecture.png')
    plt.savefig(arch_img_path, bbox_inches='tight', dpi=200)
    plt.close()
    return arch_img_path

def create_comprehensive_plots(trainer, results, report_dir):
    """Create comprehensive visualization plots (moved from ModelTrainer)"""
    # ...existing code from ModelTrainer.create_comprehensive_plots...
    # Use trainer.history, trainer.training_stats, etc.
    # Copy the full method body from train_model.py
    pass

def generate_pdf_report(trainer, params, results, model_path, version, report_dir):
    """Generate comprehensive PDF report (moved from ModelTrainer)"""
    pdf_path = os.path.join(report_dir, f'complete_training_report_v{version}.pdf')
    with PdfPages(pdf_path) as pdf:
        # ...existing code for cover page and other pages...
        # Insert neural network architecture diagram page
        arch_img_path = os.path.join(report_dir, 'nn_architecture.png')
        if not os.path.exists(arch_img_path):
            arch_img_path = plot_model_architecture(trainer, report_dir)
        if os.path.exists(arch_img_path):
            fig = plt.figure(figsize=(11, 4))
            img = plt.imread(arch_img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Neural Network Architecture', fontsize=14, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        # ...existing code for detailed text report and other pages...
    return pdf_path

def generate_detailed_text_report(trainer, params, results, model_path, version, report_dir):
    """Generate detailed text report (moved from ModelTrainer)"""
    report_path = os.path.join(report_dir, 'detailed_report_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        # ...existing code for summary and model architecture...
        model_info = trainer.training_stats['model']
        model = trainer.model
        activations = []
        for layer in model.layers:
            if hasattr(layer, 'activation'):
                act = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                activations.append(f"{layer.name}: {act}")
        f.write("Activation Functions (per layer):\n")
        for act in activations:
            f.write(f"  {act}\n")
        f.write(f"\nModel Summary (Keras):\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        # ...rest of the report...
    return report_path

def export_data_to_csv(trainer, results, report_dir):
    """Export training data and results to CSV files (moved from ModelTrainer)"""
    # ...existing code from ModelTrainer.export_data_to_csv...
    pass

def generate_report(trainer, params, results, model_path, version):
    """Generate comprehensive training report with plots and PDF (moved from ModelTrainer)"""
    logger.info("Generating comprehensive training report...")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(reports_dir, f"training_report_v{version}_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    create_comprehensive_plots(trainer, results, report_dir)
    generate_detailed_text_report(trainer, params, results, model_path, version, report_dir)
    pdf_path = generate_pdf_report(trainer, params, results, model_path, version, report_dir)
    export_data_to_csv(trainer, results, report_dir)
    logger.info(f"Comprehensive training report generated in: {report_dir}")
    return report_dir, pdf_path
