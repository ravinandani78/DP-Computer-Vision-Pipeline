# python3 src/DpTraining/train.py --config configs/config.yaml
import mlflow
import os
import shutil
from ultralytics import YOLO

def train_detection(config):
    """
    Trains a YOLOv8 object detection model.
    """
    with mlflow.start_run(run_name="object_detection_training") as run:
        mlflow.log_param("model_type", "object_detection")
        mlflow.log_params(config)

        # Construct model name from config
        model_version = config['model']['version']
        model_size = config['model']['size']
        model_name = f"{model_version}{model_size}.pt"
        
        mlflow.log_param("model_name", model_name)

        # Load a model
        model = YOLO(model_name)  # load a pretrained model (recommended for training)

        # Define the project directory from config
        project_dir = config["save_dir"]

        # If the project directory exists, remove it to ensure a clean training environment
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

        # Get save_every from config, default to -1 if not present or null
        save_period = config.get("save_every")
        if save_period is None:
            save_period = -1

        # Train the model
        results = model.train(
            data=config["data_path"],
            epochs=config["epochs"],
            batch=config["batch_size"],
            imgsz=config["img_size"],
            lr0=config["learning_rate"],
            project=project_dir,
            name="detection_run",
            save_period=save_period
        )

        # Log metrics
        mlflow.log_metrics({
            "mAP50-95": results.box.map,
            "mAP50": results.box.map50,
            "mAP75": results.box.map75,
        })

        # Log model artifact
        save_dir = model.trainer.save_dir
        mlflow.log_artifacts(str(save_dir), artifact_path="yolo_detection_model")

        print(f"Object detection training complete. Model and metrics logged to MLflow run: {run.info.run_id}")
