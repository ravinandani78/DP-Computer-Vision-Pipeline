"""
Utility functions for the evaluation module.
"""

import os
import logging
from typing import Tuple, Optional

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the evaluation module.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("DpEvaluation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def detect_data_type(unseen_data_path: str) -> Optional[str]:
    """
    Automatically detect whether the unseen data corresponds to object detection or classification
    based on the folder structure.
    
    Args:
        unseen_data_path: Path to the unseen data directory
        
    Returns:
        'object_detection' if images/ and labels/ folders are found
        'classification' if numbered folders (0, 1, 2, etc.) are found
        None if neither structure is detected
    """
    logger = setup_logging()
    
    if not os.path.exists(unseen_data_path):
        logger.error(f"Unseen data path does not exist: {unseen_data_path}")
        return None
    
    if not os.path.isdir(unseen_data_path):
        logger.error(f"Unseen data path is not a directory: {unseen_data_path}")
        return None
    
    contents = os.listdir(unseen_data_path)
    
    # Check for object detection structure (images/ and labels/ folders)
    if 'images' in contents and 'labels' in contents:
        images_path = os.path.join(unseen_data_path, 'images')
        labels_path = os.path.join(unseen_data_path, 'labels')
        
        if os.path.isdir(images_path) and os.path.isdir(labels_path):
            logger.info("Detected object detection data structure (images/ and labels/ folders)")
            return 'object_detection'
    
    # Check for classification structure (numbered folders: 0, 1, 2, etc.)
    numeric_folders = []
    for item in contents:
        item_path = os.path.join(unseen_data_path, item)
        if os.path.isdir(item_path):
            try:
                # Check if folder name is a number
                int(item)
                numeric_folders.append(item)
            except ValueError:
                continue
    
    if len(numeric_folders) >= 1:  # At least 1 class folder
        logger.info(f"Detected classification data structure ({len(numeric_folders)} class folders)")
        return 'classification'
    
    # Check for classification structure (single folder with mixed class images)
    # Look for image files with class labels in filenames (e.g., image_0.0.jpg, image_1.0.jpg)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in contents if os.path.splitext(f.lower())[1] in image_extensions]
    
    if len(image_files) > 0:
        # Check if images have class labels in filenames
        class_labels_found = set()
        for img_file in image_files[:5]:  # Check first 5 files
            # Look for pattern like "_X.X.jpg" where X.X is the class label
            if '_' in img_file:
                parts = img_file.split('_')
                if len(parts) >= 2:
                    # Get the last part before extension
                    last_part = os.path.splitext(parts[-1])[0]
                    try:
                        # Try to parse as float (class label)
                        class_label = float(last_part)
                        class_labels_found.add(int(class_label))  # Convert to int for class
                    except ValueError:
                        continue
        
        if len(class_labels_found) > 0:
            logger.info(f"Detected classification data structure (single folder with {len(image_files)} images, {len(class_labels_found)} classes detected)")
            return 'classification'
    
    logger.warning(f"Could not detect data type for path: {unseen_data_path}")
    logger.warning(f"Contents: {contents}")
    return None

def validate_config_paths(config: dict) -> Tuple[bool, str]:
    """
    Validate that all required paths in the config exist and are accessible.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    logger = setup_logging()
    
    # Check if evaluation section exists
    if 'evaluation' not in config:
        return False, "Missing 'evaluation' section in config"
    
    eval_config = config['evaluation']
    
    # Check object detection config
    if 'object_detection' in eval_config and eval_config['object_detection'].get('enabled', False):
        od_config = eval_config['object_detection']
        
        if 'unseen_data_path' not in od_config or not od_config['unseen_data_path']:
            return False, "Missing 'unseen_data_path' for object detection evaluation"
        
        if not os.path.exists(od_config['unseen_data_path']):
            return False, f"Object detection unseen_data_path does not exist: {od_config['unseen_data_path']}"
    
    # Check classification config
    if 'classification' in eval_config and eval_config['classification'].get('enabled', False):
        cls_config = eval_config['classification']
        
        if 'unseen_data_path' not in cls_config or not cls_config['unseen_data_path']:
            return False, "Missing 'unseen_data_path' for classification evaluation"
        
        if not os.path.exists(cls_config['unseen_data_path']):
            return False, f"Classification unseen_data_path does not exist: {cls_config['unseen_data_path']}"
    
    return True, ""

def create_output_directory(save_dir: str, task_type: str) -> str:
    """
    Create output directory for evaluation results.
    
    Args:
        save_dir: Base save directory
        task_type: Type of task ('object_detection' or 'classification')
        
    Returns:
        Full path to the created output directory
    """
    logger = setup_logging()
    
    if task_type == 'object_detection':
        output_dir = os.path.join(save_dir, 'object_detection_results')
    elif task_type == 'classification':
        output_dir = os.path.join(save_dir, 'classification_results')
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    return output_dir
