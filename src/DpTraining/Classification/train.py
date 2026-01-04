import mlflow
import os
import shutil
from ultralytics import YOLO

def train_classification(config):
    """
    Trains a YOLOv8 classification model.
    """
    with mlflow.start_run(run_name="classification_training") as run:
        mlflow.log_param("model_type", "classification")
        mlflow.log_params(config)

        # Construct model name from config
        model_version = config['model']['version']
        model_size = config['model']['size']
        model_name = f"{model_version}{model_size}-cls.pt"
        
        mlflow.log_param("model_name", model_name)

        # Load a model
        model = YOLO(model_name)  # load a pretrained model

        # Define the project directory from config
        project_dir = config["save_dir"]

        # If the project directory exists, remove it to ensure a clean training environment
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        os.makedirs(project_dir, exist_ok=True)

        # Get save_every from config, default to -1 if not present or null
        save_period = config.get("save_every")
        if save_period is None:
            save_period = -1

        # Train the model
        results = model.train(
            data=os.path.dirname(config["data_path"]),
            epochs=config["epochs"],
            batch=config["batch_size"],
            imgsz=config["img_size"],
            lr0=config["learning_rate"],
            project=project_dir,
            name="classification_run",
            save_period=save_period
        )

        # Log metrics
        mlflow.log_metrics({
            "accuracy_top1": results.top1,
            "accuracy_top5": results.top5,
        })

        # Log model artifact
        save_dir = model.trainer.save_dir
        mlflow.log_artifacts(str(save_dir), artifact_path="yolo_classification_model")

        print(f"Classification training complete. Model and metrics logged to MLflow run: {run.info.run_id}")
