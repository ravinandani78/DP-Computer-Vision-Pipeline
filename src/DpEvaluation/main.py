# python3 src/DpEvaluation/main.py --config configs/config.yaml
import yaml
import argparse
import os
import sys

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DpEvaluation.ObjectDetection.evaluate import evaluate_detection
from DpEvaluation.Classification.evaluate import evaluate_classification
from DpEvaluation.utils import setup_logging, validate_config_paths, detect_data_type

def main(config_path):
    """
    Main function to run the config-driven evaluation pipeline.
    Automatically detects data types and runs appropriate evaluations.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    logger = setup_logging()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate configuration
        is_valid, error_msg = validate_config_paths(config)
        if not is_valid:
            logger.error(f"Configuration validation failed: {error_msg}")
            return

        evaluation_config = config.get('evaluation', {})
        
        # Check if any evaluation is enabled
        od_enabled = evaluation_config.get('object_detection', {}).get('enabled', False)
        cls_enabled = evaluation_config.get('classification', {}).get('enabled', False)
        
        if not od_enabled and not cls_enabled:
            logger.warning("No evaluation modules are enabled in the configuration")
            return

        # Object Detection Evaluation
        if od_enabled:
            logger.info("=" * 60)
            logger.info("Starting Object Detection Evaluation...")
            logger.info("=" * 60)
            
            od_config = evaluation_config['object_detection']
            unseen_data_path = od_config.get('unseen_data_path')
            
            if unseen_data_path and os.path.exists(unseen_data_path):
                # Detect data type to ensure it matches expectation
                detected_type = detect_data_type(unseen_data_path)
                if detected_type == 'object_detection':
                    evaluate_detection(config)
                    logger.info("Object Detection Evaluation completed successfully")
                else:
                    logger.error(f"Expected object detection data structure, but found: {detected_type}")
            else:
                logger.error(f"Object detection unseen_data_path not found: {unseen_data_path}")
        
        # Classification Evaluation  
        if cls_enabled:
            logger.info("=" * 60)
            logger.info("Starting Classification Evaluation...")
            logger.info("=" * 60)
            
            cls_config = evaluation_config['classification']
            unseen_data_path = cls_config.get('unseen_data_path')
            
            if unseen_data_path and os.path.exists(unseen_data_path):
                # Detect data type to ensure it matches expectation
                detected_type = detect_data_type(unseen_data_path)
                if detected_type == 'classification':
                    evaluate_classification(config)
                    logger.info("Classification Evaluation completed successfully")
                else:
                    logger.error(f"Expected classification data structure, but found: {detected_type}")
            else:
                logger.error(f"Classification unseen_data_path not found: {unseen_data_path}")

        logger.info("=" * 60)
        logger.info("Evaluation pipeline completed")
        logger.info("=" * 60)
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the config-driven evaluation pipeline with automatic data type detection."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the config.yaml file containing evaluation parameters.'
    )
    args = parser.parse_args()
    main(args.config)
