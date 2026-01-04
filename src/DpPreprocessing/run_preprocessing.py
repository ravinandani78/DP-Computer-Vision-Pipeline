# python3 src/DpPreprocessing/run_preprocessing.py --config configs/config.yaml
"""
This Module handles the Data Preparation, The Entire Functionality is handled with the Config File.
Please check the config/config_data_preparation.py
python3 src/DpPreprocessing/run_preprocessing.py --config configs/config.yaml
"""

# imports 
import mlflow
import yaml
import argparse
import os
import sys

# Add src to path to allow imports from DpPreprocessing etc.
# Since we're now inside DpPreprocessing, we need to go up one level to reach src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the same directory (DpPreprocessing)
from DpPreprocessing.main import PrepareDataset

def main(config_path):
    """
    Main function to run the data preparation pipeline.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with mlflow.start_run(run_name="preprocessing") as run:
        mlflow.log_param("raw_data_dir", config['data']['raw_data_dir'])
        mlflow.log_param("processed_dir", config['data']['processed_dir'])
        mlflow.log_params(config['preprocessing'])

        prep = PrepareDataset(config)
        prep()

        processed_data_dir = config['data']['processed_dir']
        if os.path.exists(processed_data_dir):
            mlflow.log_artifacts(processed_data_dir, artifact_path="preprocessed_data")
            print(f"Preprocessing complete. Artifacts logged to MLflow run: {run.info.run_id}")
        else:
            print(f"Processed data directory not found: {processed_data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLflow preprocessing pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file.')
    args = parser.parse_args()
    main(args.config)
