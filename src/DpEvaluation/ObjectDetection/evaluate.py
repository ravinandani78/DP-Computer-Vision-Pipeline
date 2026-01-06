
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
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging, detect_data_type, create_output_directory

def evaluate_detection(config):
    """
    Evaluates the object detection model using config-driven approach.
    Automatically detects data structure and computes comprehensive metrics.
    
    Args:
        config: Configuration dictionary containing evaluation parameters
    """
    logger = setup_logging()
    
    run_id = None
    try:
        with mlflow.start_run(run_name="object_detection_evaluation") as run:
            # Store run_id early to avoid accessing it after context exits
            run_id = run.info.run_id
            
            mlflow.log_param("model_type", "object_detection_evaluation")
            
            eval_config = config['evaluation']['object_detection']
            train_config = config['training']['object_detection']
            mlflow.log_params(eval_config)

            # Load the trained model
            model_weights = eval_config.get('model_weights')
            if not model_weights or not os.path.exists(model_weights):
                training_save_dir = train_config['save_dir']
                try:
                    # The training saves runs in a subdirectory, e.g., 'detection_run'
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
                logger.error("Missing 'unseen_data_path' in object detection evaluation config")
                return
            
            if not os.path.exists(unseen_data_path):
                logger.error(f"Unseen data path does not exist: {unseen_data_path}")
                return

            # Detect data type
            data_type = detect_data_type(unseen_data_path)
            if data_type != 'object_detection':
                logger.error(f"Expected object detection data structure, but detected: {data_type}")
                return

            # Create output directory
            base_save_dir = eval_config['save_dir']
            output_dir = create_output_directory(base_save_dir, 'object_detection')
            
            mlflow.log_param("evaluation_data_path", unseen_data_path)
            mlflow.log_param("output_directory", output_dir)

            logger.info(f"Running evaluation on data: {unseen_data_path}")
            logger.info(f"Saving results to: {output_dir}")

            # Create temporary data.yaml for evaluation
            temp_yaml_path = os.path.join(output_dir, 'temp_eval_data.yaml')
            _create_temp_data_yaml(unseen_data_path, temp_yaml_path, train_config)

            # Run evaluation using YOLO's built-in validation
            logger.info("Running YOLO validation...")
            metrics = model.val(data=temp_yaml_path,
                               split='val',
                               project=os.path.dirname(output_dir),
                               name=os.path.basename(output_dir),
                               save_json=True)

            # Extract and log comprehensive metrics
            metrics_dict = _extract_detection_metrics(metrics)
            mlflow.log_metrics(metrics_dict)

            # Generate additional evaluation reports
            _generate_detection_reports(model, unseen_data_path, output_dir, metrics)

            # Save metrics to JSON file
            metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)

            # Log artifacts to MLflow
            mlflow.log_artifacts(output_dir, artifact_path="evaluation_results")

            # Clean up temporary files
            if os.path.exists(temp_yaml_path):
                os.remove(temp_yaml_path)

            logger.info(f"Evaluation complete. Results saved in {output_dir} and logged to MLflow run: {run_id}")
            
    except Exception as e:
        if run_id:
            logger.error(f"Error during object detection evaluation. MLflow run ID: {run_id}. Error: {str(e)}")
        else:
            logger.error(f"Error during object detection evaluation: {str(e)}")
        raise

def _create_temp_data_yaml(unseen_data_path, temp_yaml_path, train_config):
    """Create a temporary data.yaml file for evaluation."""
    logger = setup_logging()
    
    # Get class names from training config
    train_data_yaml = train_config['data_path']
    with open(train_data_yaml, 'r') as f:
        train_data_content = yaml.safe_load(f)
    
    # Create evaluation data yaml - YOLO requires train and val keys even for evaluation
    # Use absolute paths to avoid YOLO scanning the entire project directory
    eval_data = {
        'path': os.path.abspath(unseen_data_path),
        'train': 'images',  # Use images subfolder
        'val': 'images',    # Use images subfolder for validation
        'test': 'images',   # Use images subfolder for testing
        'names': train_data_content.get('names', {}),
        'nc': train_data_content.get('nc', len(train_data_content.get('names', {})))
    }
    
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(eval_data, f)
    
    logger.info(f"Created temporary data.yaml: {temp_yaml_path}")

def _extract_detection_metrics(metrics):
    """Extract comprehensive metrics from YOLO validation results."""
    metrics_dict = {}
    
    if hasattr(metrics, 'box'):
        box_metrics = metrics.box
        
        # mAP metrics
        metrics_dict['mAP50-95'] = float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0
        metrics_dict['mAP50'] = float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0
        metrics_dict['mAP75'] = float(box_metrics.map75) if hasattr(box_metrics, 'map75') else 0.0
        
        # Precision and Recall
        if hasattr(box_metrics, 'mp'):
            metrics_dict['mean_precision'] = float(box_metrics.mp)
        if hasattr(box_metrics, 'mr'):
            metrics_dict['mean_recall'] = float(box_metrics.mr)
        
        # Per-class metrics if available
        if hasattr(box_metrics, 'ap'):
            ap_per_class = box_metrics.ap
            if ap_per_class is not None:
                try:
                    # Handle case where ap_per_class is an array of arrays
                    if hasattr(ap_per_class, '__len__') and len(ap_per_class) > 0:
                        for i, ap in enumerate(ap_per_class):
                            if hasattr(ap, '__len__') and len(ap) > 0:
                                metrics_dict[f'class_{i}_mAP50-95'] = float(np.mean(ap))
                            elif isinstance(ap, (int, float, np.number)):
                                metrics_dict[f'class_{i}_mAP50-95'] = float(ap)
                except (TypeError, AttributeError):
                    # Handle case where ap_per_class is a single value
                    if isinstance(ap_per_class, (int, float, np.number)):
                        metrics_dict['class_0_mAP50-95'] = float(ap_per_class)
        
        if hasattr(box_metrics, 'ap50'):
            ap50_per_class = box_metrics.ap50
            if ap50_per_class is not None:
                try:
                    if hasattr(ap50_per_class, '__len__') and len(ap50_per_class) > 0:
                        for i, ap50 in enumerate(ap50_per_class):
                            if isinstance(ap50, (int, float, np.number)):
                                metrics_dict[f'class_{i}_mAP50'] = float(ap50)
                except (TypeError, AttributeError):
                    # Handle case where ap50_per_class is a single value
                    if isinstance(ap50_per_class, (int, float, np.number)):
                        metrics_dict['class_0_mAP50'] = float(ap50_per_class)
    
    # Speed metrics
    if hasattr(metrics, 'speed'):
        speed = metrics.speed
        if isinstance(speed, dict):
            for key, value in speed.items():
                metrics_dict[f'speed_{key}'] = float(value)
    
    return metrics_dict

def _generate_detection_reports(model, unseen_data_path, output_dir, metrics):
    """Generate additional evaluation reports and visualizations."""
    logger = setup_logging()
    
    try:
        # Generate confusion matrix if available
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            cm = metrics.confusion_matrix.matrix
            if cm is not None:
                plt.figure(figsize=(10, 8))
                # Convert to integers for proper formatting
                cm_int = cm.astype(int)
                sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated confusion matrix plot")
        
        # Create evaluation summary report
        _create_evaluation_summary(output_dir, metrics)
        
    except Exception as e:
        logger.warning(f"Could not generate additional reports: {str(e)}")

def _create_evaluation_summary(output_dir, metrics):
    """Create a comprehensive evaluation summary report."""
    logger = setup_logging()
    
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("Object Detection Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        if hasattr(metrics, 'box'):
            box_metrics = metrics.box
            f.write("Box Detection Metrics:\n")
            f.write("-" * 25 + "\n")
            
            if hasattr(box_metrics, 'map'):
                f.write(f"mAP (0.5:0.95): {box_metrics.map:.4f}\n")
            if hasattr(box_metrics, 'map50'):
                f.write(f"mAP (0.5): {box_metrics.map50:.4f}\n")
            if hasattr(box_metrics, 'map75'):
                f.write(f"mAP (0.75): {box_metrics.map75:.4f}\n")
            if hasattr(box_metrics, 'mp'):
                f.write(f"Mean Precision: {box_metrics.mp:.4f}\n")
            if hasattr(box_metrics, 'mr'):
                f.write(f"Mean Recall: {box_metrics.mr:.4f}\n")
        
        f.write("\n")
        
        if hasattr(metrics, 'speed'):
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            speed = metrics.speed
            if isinstance(speed, dict):
                for key, value in speed.items():
                    f.write(f"{key}: {value:.2f} ms\n")
    
    logger.info(f"Created evaluation summary: {summary_path}")
