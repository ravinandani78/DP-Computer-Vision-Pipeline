import mlflow
import os
import shutil
from ultralytics import YOLO

def train_classification(config):
    """
    Trains a YOLOv8 classification model.
    """
    # Ensure MLflow tracking URI is set to the default location
    # Go up from Classification/train.py -> DpTraining -> src -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    mlruns_dir = os.path.join(project_root, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    run_id = None
    run = None
    training_successful = False
    try:
        run = mlflow.start_run(run_name="classification_training")
        # Store run_id early to avoid accessing it after context exits
        run_id = run.info.run_id
        
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

        training_successful = True
        print(f"Classification training complete. Model and metrics logged to MLflow run: {run_id}")
        
    except Exception as e:
        if run_id:
            print(f"Classification training encountered an error. MLflow run ID: {run_id}")
        else:
            print(f"Classification training encountered an error before MLflow run was created.")
        raise
    finally:
        # Manually end the run to avoid context manager issues
        if run is not None:
            try:
                status = "FINISHED" if training_successful else "FAILED"
                mlflow.end_run(status=status)
            except Exception as e:
                # Suppress errors during run finalization - run data is already saved
                # This is a known issue with MLflow file store when runs are finalized
                print(f"Warning: Could not finalize MLflow run (this is usually safe to ignore): {str(e)}")
