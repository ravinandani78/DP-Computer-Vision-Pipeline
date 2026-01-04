from ultralytics import YOLO

# Load a pretrained YOLOv8n model


def predict(weights_path, source, name, conf, img_size, project_name):
    model = YOLO(weights_path)
    # Run inference on 'bus.jpg' with arguments
    model.predict(
        source,
        save=True,
        imgsz=int(img_size),
        conf=float(conf),
        )

    