# python3 src/DpTraining/train.py --config configs/config.yaml
import mlflow
import os
import shutil
import json
import time

# #region agent log
with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
    f.write(json.dumps({"sessionId":"debug-session","runId":"import","hypothesisId":"A","location":"ObjectDetection/train.py:7","message":"Before YOLO import","data":{"yolo_mlflow_disabled":os.environ.get("YOLO_MLFLOW_LOGGING_DISABLED"),"mlflow_tracking_uri":mlflow.get_tracking_uri()},"timestamp":int(time.time()*1000)}) + '\n')
# #endregion

from ultralytics import YOLO

# #region agent log
with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
    f.write(json.dumps({"sessionId":"debug-session","runId":"import","hypothesisId":"A","location":"ObjectDetection/train.py:11","message":"After YOLO import","data":{"yolo_mlflow_disabled":os.environ.get("YOLO_MLFLOW_LOGGING_DISABLED")},"timestamp":int(time.time()*1000)}) + '\n')
# #endregion

def train_detection(config):
    """
    Trains a YOLOv8 object detection model.
    """
    # Ensure MLflow tracking URI is set to the default location
    # Go up from ObjectDetection/train.py -> DpTraining -> src -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    mlruns_dir = os.path.join(project_root, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    # #region agent log
    with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-run-fix","hypothesisId":"C","location":"ObjectDetection/train.py:27","message":"Fixed tracking URI","data":{"tracking_uri":mlflow.get_tracking_uri(),"mlruns_dir":mlruns_dir,"project_root":project_root},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    
    # #region agent log
    with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-run","hypothesisId":"C","location":"ObjectDetection/train.py:18","message":"MLflow tracking URI set","data":{"tracking_uri":mlflow.get_tracking_uri(),"mlruns_dir":mlruns_dir},"timestamp":int(time.time()*1000)}) + '\n')
    # #endregion
    
    run_id = None
    run = None
    training_successful = False
    try:
        run = mlflow.start_run(run_name="object_detection_training")
        # Store run_id early to avoid accessing it after context exits
        run_id = run.info.run_id
        
        # #region agent log
        with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-run","hypothesisId":"C","location":"ObjectDetection/train.py:25","message":"MLflow run created","data":{"run_id":run_id,"run_name":run.info.run_name},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        
        # #region agent log
        meta_file = os.path.join(mlruns_dir, "0", run_id, "meta.yaml")
        meta_exists = os.path.exists(meta_file)
        with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-run","hypothesisId":"D","location":"ObjectDetection/train.py:30","message":"Check run metadata file","data":{"run_id":run_id,"meta_file":meta_file,"meta_exists":meta_exists},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        
        mlflow.log_param("model_type", "object_detection")
        mlflow.log_params(config)

        # Construct model name from config
        model_version = config['model']['version']
        model_size = config['model']['size']
        model_name = f"{model_version}{model_size}.pt"
        
        mlflow.log_param("model_name", model_name)
        
        # Disable Ultralytics MLflow callback BEFORE creating model
        # #region agent log
        try:
            from ultralytics.utils.callbacks import mlflow as yolo_mlflow
            callback_enabled_before = hasattr(yolo_mlflow, 'on_train_epoch_end') and callable(yolo_mlflow.on_train_epoch_end)
            with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:64","message":"Before disabling callback","data":{"run_id":run_id,"callback_enabled":callback_enabled_before},"timestamp":int(time.time()*1000)}) + '\n')
        except Exception as e:
            with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:64","message":"Error checking callback","data":{"run_id":run_id,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        
        # Disable the callback by replacing it with a no-op
        # Also need to prevent the callback from being registered in the trainer's callbacks list
        try:
            from ultralytics.utils.callbacks import mlflow as yolo_mlflow
            def disabled_callback(*args, **kwargs):
                # #region agent log
                with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"callback-called","hypothesisId":"B","location":"ObjectDetection/train.py:disabled_callback","message":"Disabled callback was called","data":{"run_id":run_id},"timestamp":int(time.time()*1000)}) + '\n')
                # #endregion
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
            # #region agent log
            with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:102","message":"After disabling callback","data":{"run_id":run_id,"callback_disabled":True,"mlflow_set_to_none":hasattr(yolo_mlflow, 'mlflow') and yolo_mlflow.mlflow is None},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
        except Exception as e:
            # #region agent log
            with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:102","message":"Error disabling callback","data":{"run_id":run_id,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
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
                    # #region agent log
                    with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:120","message":"Removed MLflow from trainer callbacks","data":{"run_id":run_id,"original_callback_count":len(original_callbacks) if isinstance(original_callbacks, dict) else 0,"new_callback_count":len(model.trainer.callbacks) if isinstance(model.trainer.callbacks, dict) else 0},"timestamp":int(time.time()*1000)}) + '\n')
                    # #endregion
        except Exception as e:
            # #region agent log
            with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:120","message":"Error removing callback from trainer","data":{"run_id":run_id,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
            # #endregion
            pass
        
        # #region agent log
        with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"B","location":"ObjectDetection/train.py:130","message":"After YOLO model creation","data":{"run_id":run_id,"has_trainer":hasattr(model,"trainer")},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion

        # Define the project directory from config
        project_dir = config["save_dir"]

        # If the project directory exists, remove it to ensure a clean training environment
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

        # Get save_every from config, default to -1 if not present or null
        save_period = config.get("save_every")
        if save_period is None:
            save_period = -1

        # #region agent log
        with open('/home/bacancy/projects/DP_Pipelines/DP-Computer-Vision-Pipeline/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-train","hypothesisId":"E","location":"ObjectDetection/train.py:54","message":"Before model.train()","data":{"run_id":run_id,"mlflow_active_run":mlflow.active_run().info.run_id if mlflow.active_run() else None,"mlflow_tracking_uri":mlflow.get_tracking_uri()},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        
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
