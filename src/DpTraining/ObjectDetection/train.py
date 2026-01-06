# python3 src/DpTraining/train.py --config configs/config.yaml
import mlflow
import os
import shutil

from ultralytics import YOLO

def train_detection(config):
    """
    Trains a YOLOv8 object detection model.
    """
    # Ensure MLflow tracking URI is set to the default location
    # Go up from ObjectDetection/train.py -> DpTraining -> src -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    mlruns_dir = os.path.join(project_root, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    run_id = None
    run = None
    training_successful = False
    try:
        run = mlflow.start_run(run_name="object_detection_training")
        # Store run_id early to avoid accessing it after context exits
        run_id = run.info.run_id
        
        mlflow.log_param("model_type", "object_detection")
        mlflow.log_params(config)

        # Construct model name from config
        model_version = config['model']['version']
        model_size = config['model']['size']
        model_name = f"{model_version}{model_size}.pt"
        
        mlflow.log_param("model_name", model_name)
        
        # Disable Ultralytics MLflow callback BEFORE creating model
        
        # Disable the callback by replacing it with a no-op
        # Also need to prevent the callback from being registered in the trainer's callbacks list
        try:
            from ultralytics.utils.callbacks import mlflow as yolo_mlflow
            def disabled_callback(*args, **kwargs):
                pass
            if hasattr(yolo_mlflow, 'on_train_epoch_end'):
                yolo_mlflow.on_train_epoch_end = disabled_callback
            if hasattr(yolo_mlflow, 'on_train_start'):
                yolo_mlflow.on_train_start = disabled_callback
            if hasattr(yolo_mlflow, 'on_train_end'):
                yolo_mlflow.on_train_end = disabled_callback
            # Also set mlflow to None in the module to prevent the callback from working
            # This is a more aggressive approach - prevent mlflow from being used
            if hasattr(yolo_mlflow, 'mlflow'):
                yolo_mlflow.mlflow = None
        except Exception as e:
            pass

        # Load a model
        model = YOLO(model_name)  # load a pretrained model (recommended for training)
        
        # Remove MLflow callback from trainer's callbacks if it exists
        try:
            if hasattr(model, 'trainer') and model.trainer is not None:
                if hasattr(model.trainer, 'callbacks'):
                    # Filter out MLflow callbacks
                    original_callbacks = model.trainer.callbacks
                    model.trainer.callbacks = {k: v for k, v in original_callbacks.items() if 'mlflow' not in k.lower()}
        except Exception as e:
            pass

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

        training_successful = True
        print(f"Object detection training complete. Model and metrics logged to MLflow run: {run_id}")
        
    except Exception as e:
        if run_id:
            print(f"Object detection training encountered an error. MLflow run ID: {run_id}")
        else:
            print(f"Object detection training encountered an error before MLflow run was created.")
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
