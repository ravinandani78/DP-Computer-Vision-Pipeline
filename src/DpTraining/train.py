# python3 src/DpTraining/train.py --config configs/config.yaml
import yaml
import argparse
import os
import sys

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DpTraining.ObjectDetection.train import train_detection
from DpTraining.Classification.train import train_classification
import argparse
import os
import sys

# Disable ultralytics's own MLflow logging at the very start
os.environ["YOLO_MLFLOW_LOGGING_DISABLED"] = "True"

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DpTraining.ObjectDetection.train import train_detection
from DpTraining.Classification.train import train_classification

def main(config_path):
    """
    Main function to run the training pipeline.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config['training']['object_detection']['enabled']:
        print("Starting Object Detection Training...")
        train_detection(config['training']['object_detection'])
    if config['training']['classification']['enabled']:
        print("Starting Classification Training...")
        train_classification(config['training']['classification'])
    else:
        print("No training module is enabled in the config file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLflow training pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file.')
    args = parser.parse_args()
    main(args.config)
