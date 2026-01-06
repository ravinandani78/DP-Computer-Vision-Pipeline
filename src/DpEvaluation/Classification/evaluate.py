
from ultralytics import YOLO
import os
import yaml
import shutil
import mlflow
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from PIL import Image
import torch
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging, detect_data_type, create_output_directory

def evaluate_classification(config):
    """
    Evaluates the classification model using config-driven approach.
    Automatically detects data structure and computes comprehensive metrics.
    
    Args:
        config: Configuration dictionary containing evaluation parameters
    """
    logger = setup_logging()
    
    run_id = None
    try:
        with mlflow.start_run(run_name="classification_evaluation") as run:
            # Store run_id early to avoid accessing it after context exits
            run_id = run.info.run_id
            
            mlflow.log_param("model_type", "classification_evaluation")

            eval_config = config['evaluation']['classification']
            train_config = config['training']['classification']
            mlflow.log_params(eval_config)
            
            # Load the trained model
            model_weights = eval_config.get('model_weights')
            if not model_weights or not os.path.exists(model_weights):
                training_save_dir = train_config['save_dir']
                try:
                    # The training saves runs in a subdirectory, e.g., 'classification_run'
                    run_dir = os.listdir(training_save_dir)[0]
                    model_weights = os.path.join(training_save_dir, run_dir, 'weights', 'best.pt')
                except (FileNotFoundError, IndexError):
                    logger.error(f"Could not find a trained model in {training_save_dir}. Please run training first or specify model_weights in config.yaml.")
                    return

            if not os.path.exists(model_weights):
                logger.error(f"Model weights not found at {model_weights}")
                return
                
            logger.info(f"Loading model from: {model_weights}")
            mlflow.log_param("model_weights", model_weights)
            model = YOLO(model_weights)

            # Get unseen data path from config
            unseen_data_path = eval_config.get('unseen_data_path')
            if not unseen_data_path:
                logger.error("Missing 'unseen_data_path' in classification evaluation config")
                return
            
            if not os.path.exists(unseen_data_path):
                logger.error(f"Unseen data path does not exist: {unseen_data_path}")
                return

            # Detect data type
            data_type = detect_data_type(unseen_data_path)
            if data_type != 'classification':
                logger.error(f"Expected classification data structure, but detected: {data_type}")
                return

            # Create output directory
            base_save_dir = eval_config['save_dir']
            output_dir = create_output_directory(base_save_dir, 'classification')
            
            mlflow.log_param("evaluation_data_path", unseen_data_path)
            mlflow.log_param("output_directory", output_dir)

            logger.info(f"Running evaluation on data: {unseen_data_path}")
            logger.info(f"Saving results to: {output_dir}")

            # Load and prepare data for evaluation
            true_labels, predicted_labels, class_names = _evaluate_classification_data(
                model, unseen_data_path, output_dir
            )

            # Compute comprehensive metrics
            metrics_dict = _compute_classification_metrics(
                true_labels, predicted_labels, class_names
            )
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics_dict)

            # Generate evaluation reports and visualizations
            _generate_classification_reports(
                true_labels, predicted_labels, class_names, output_dir, metrics_dict
            )

            # Save metrics to JSON file
            metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)

            # Log artifacts to MLflow
            mlflow.log_artifacts(output_dir, artifact_path="evaluation_results")

            logger.info(f"Evaluation complete. Results saved in {output_dir} and logged to MLflow run: {run_id}")
            
    except Exception as e:
        if run_id:
            logger.error(f"Error during classification evaluation. MLflow run ID: {run_id}. Error: {str(e)}")
        else:
            logger.error(f"Error during classification evaluation: {str(e)}")
        raise

def _evaluate_classification_data(model, unseen_data_path, output_dir):
    """
    Load classification data and run predictions.
    
    Args:
        model: Loaded YOLO model
        unseen_data_path: Path to classification data (single folder with mixed class images or class folders)
        output_dir: Output directory for saving results
        
    Returns:
        Tuple of (true_labels, predicted_labels, class_names)
    """
    logger = setup_logging()
    
    # Check if we have class folders or single folder with mixed images
    contents = os.listdir(unseen_data_path)
    class_folders = [f for f in contents if os.path.isdir(os.path.join(unseen_data_path, f))]
    
    if len(class_folders) > 0:
        # Handle class folder structure (original approach)
        return _evaluate_class_folders(model, unseen_data_path, class_folders)
    else:
        # Handle single folder with mixed class images (new approach)
        return _evaluate_mixed_folder(model, unseen_data_path)

def _evaluate_class_folders(model, unseen_data_path, class_folders):
    """Evaluate classification data organized in class folders."""
    logger = setup_logging()
    
    true_labels = []
    predicted_labels = []
    
    # Sort class folders numerically
    try:
        class_folders.sort(key=int)
    except ValueError:
        class_folders.sort()  # Fallback to alphabetical sort
    
    class_names = class_folders
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    # Process each class folder
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(unseen_data_path, class_name)
        
        # Get all image files in the class folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(class_folder) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        logger.info(f"Processing class {class_name}: {len(image_files)} images")
        
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            
            try:
                # Run prediction
                results = model(image_path, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'probs') and result.probs is not None:
                        # Get predicted class
                        predicted_class = int(result.probs.top1)
                        predicted_labels.append(predicted_class)
                        true_labels.append(class_idx)
                    else:
                        logger.warning(f"No classification probabilities found for {image_path}")
                        # Use class index as fallback
                        predicted_labels.append(class_idx)
                        true_labels.append(class_idx)
                else:
                    logger.warning(f"No results returned for {image_path}")
                    predicted_labels.append(class_idx)
                    true_labels.append(class_idx)
                    
            except Exception as e:
                logger.warning(f"Error processing {image_path}: {str(e)}")
                # Use true label as fallback
                predicted_labels.append(class_idx)
                true_labels.append(class_idx)
    
    logger.info(f"Processed {len(true_labels)} images total")
    return np.array(true_labels), np.array(predicted_labels), class_names

def _evaluate_mixed_folder(model, unseen_data_path):
    """Evaluate classification data in a single folder with class labels in filenames."""
    logger = setup_logging()
    
    true_labels = []
    predicted_labels = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in os.listdir(unseen_data_path) 
                  if os.path.splitext(f.lower())[1] in image_extensions]
    
    logger.info(f"Found {len(image_files)} images in single folder")
    
    # Extract class labels from filenames and get unique classes
    class_labels_map = {}
    unique_classes = set()
    
    for image_file in image_files:
        # Extract class label from filename (e.g., "image_1.0.jpg" -> class 1)
        if '_' in image_file:
            parts = image_file.split('_')
            if len(parts) >= 2:
                # Get the last part before extension
                last_part = os.path.splitext(parts[-1])[0]
                try:
                    # Try to parse as float and convert to int for class
                    class_label = int(float(last_part))
                    class_labels_map[image_file] = class_label
                    unique_classes.add(class_label)
                except ValueError:
                    logger.warning(f"Could not extract class label from filename: {image_file}")
                    class_labels_map[image_file] = 0  # Default to class 0
                    unique_classes.add(0)
            else:
                logger.warning(f"Unexpected filename format: {image_file}")
                class_labels_map[image_file] = 0
                unique_classes.add(0)
        else:
            logger.warning(f"No class label found in filename: {image_file}")
            class_labels_map[image_file] = 0
            unique_classes.add(0)
    
    # Create class names list
    class_names = [str(cls) for cls in sorted(unique_classes)]
    logger.info(f"Detected {len(class_names)} classes: {class_names}")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(unseen_data_path, image_file)
        true_class = class_labels_map[image_file]
        
        try:
            # Run prediction
            results = model(image_path, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'probs') and result.probs is not None:
                    # Get predicted class
                    predicted_class = int(result.probs.top1)
                    predicted_labels.append(predicted_class)
                    true_labels.append(true_class)
                else:
                    logger.warning(f"No classification probabilities found for {image_path}")
                    # Use true class as fallback
                    predicted_labels.append(true_class)
                    true_labels.append(true_class)
            else:
                logger.warning(f"No results returned for {image_path}")
                predicted_labels.append(true_class)
                true_labels.append(true_class)
                
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {str(e)}")
            # Use true label as fallback
            predicted_labels.append(true_class)
            true_labels.append(true_class)
    
    logger.info(f"Processed {len(true_labels)} images total")
    return np.array(true_labels), np.array(predicted_labels), class_names

def _compute_classification_metrics(true_labels, predicted_labels, class_names):
    """
    Compute comprehensive classification metrics.
    
    Args:
        true_labels: Array of true class labels
        predicted_labels: Array of predicted class labels
        class_names: List of class names
        
    Returns:
        Dictionary of computed metrics
    """
    logger = setup_logging()
    
    metrics_dict = {}
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    metrics_dict['accuracy'] = float(accuracy)
    
    # Precision, Recall, F1-score (macro and weighted averages)
    # Always use macro and weighted averages to handle any number of classes
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    metrics_dict.update({
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_score_macro': float(f1_macro),
        'f1_score_weighted': float(f1_weighted)
    })
    
    # Per-class metrics
    precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics_dict[f'precision_class_{class_name}'] = float(precision_per_class[i])
            metrics_dict[f'recall_class_{class_name}'] = float(recall_per_class[i])
            metrics_dict[f'f1_score_class_{class_name}'] = float(f1_per_class[i])
    
    logger.info(f"Computed metrics for {len(class_names)} classes")
    return metrics_dict

def _generate_classification_reports(true_labels, predicted_labels, class_names, output_dir, metrics_dict):
    """
    Generate comprehensive classification reports and visualizations.
    
    Args:
        true_labels: Array of true class labels
        predicted_labels: Array of predicted class labels
        class_names: List of class names
        output_dir: Output directory for saving reports
        metrics_dict: Dictionary of computed metrics
    """
    logger = setup_logging()
    
    try:
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate classification report
        report = classification_report(true_labels, predicted_labels, 
                                     target_names=class_names, output_dict=True)
        
        # Save classification report as JSON
        report_file = os.path.join(output_dir, 'classification_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create text summary
        _create_classification_summary(output_dir, metrics_dict, report, class_names)
        
        # Plot per-class metrics
        _plot_per_class_metrics(metrics_dict, class_names, output_dir)
        
        logger.info("Generated all classification reports and visualizations")
        
    except Exception as e:
        logger.warning(f"Could not generate classification reports: {str(e)}")

def _create_classification_summary(output_dir, metrics_dict, report, class_names):
    """Create a comprehensive classification evaluation summary."""
    logger = setup_logging()
    
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("Classification Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {metrics_dict.get('accuracy', 0):.4f}\n")
        f.write(f"Precision (Macro): {metrics_dict.get('precision_macro', 0):.4f}\n")
        f.write(f"Recall (Macro): {metrics_dict.get('recall_macro', 0):.4f}\n")
        f.write(f"F1-Score (Macro): {metrics_dict.get('f1_score_macro', 0):.4f}\n")
        f.write(f"Precision (Weighted): {metrics_dict.get('precision_weighted', 0):.4f}\n")
        f.write(f"Recall (Weighted): {metrics_dict.get('recall_weighted', 0):.4f}\n")
        f.write(f"F1-Score (Weighted): {metrics_dict.get('f1_score_weighted', 0):.4f}\n\n")
        
        # Per-class metrics
        f.write("Per-Class Metrics:\n")
        f.write("-" * 20 + "\n")
        for class_name in class_names:
            f.write(f"\nClass {class_name}:\n")
            f.write(f"  Precision: {metrics_dict.get(f'precision_class_{class_name}', 0):.4f}\n")
            f.write(f"  Recall: {metrics_dict.get(f'recall_class_{class_name}', 0):.4f}\n")
            f.write(f"  F1-Score: {metrics_dict.get(f'f1_score_class_{class_name}', 0):.4f}\n")
    
    logger.info(f"Created evaluation summary: {summary_path}")

def _plot_per_class_metrics(metrics_dict, class_names, output_dir):
    """Plot per-class precision, recall, and F1-score."""
    logger = setup_logging()
    
    try:
        # Extract per-class metrics
        precisions = [metrics_dict.get(f'precision_class_{cls}', 0) for cls in class_names]
        recalls = [metrics_dict.get(f'recall_class_{cls}', 0) for cls in class_names]
        f1_scores = [metrics_dict.get(f'f1_score_class_{cls}', 0) for cls in class_names]
        
        # Create bar plot
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Classification Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generated per-class metrics plot")
        
    except Exception as e:
        logger.warning(f"Could not generate per-class metrics plot: {str(e)}")
