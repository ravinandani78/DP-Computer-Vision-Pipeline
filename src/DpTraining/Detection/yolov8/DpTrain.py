import yaml
import os
import torch
from ultralytics import YOLO


LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
DROPOUT = 0.0
EPOCHS = 3000
BATCH  = 16
IMAGE_SIZE = 640


model_name = 'yolov8n.pt'
data_path = '/content/datasets/data.yaml'


        
def add_path_to_yaml_file(data_yaml):
    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)
    
    data['path'] = os.path.dirname(data_yaml)
    
    with open(data_yaml, 'w') as file:
        yaml.dump(data, file, default_flow_style = False, sort_keys=False)


def Train(cfg):    
    model = YOLO(cfg.MODELNAME)  # load a pretrained model (recommended for training)
    add_path_to_yaml_file(cfg.DATA_YAML)
    device = cfg.DEVICE
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.train(
        data=cfg.DATA_YAML, # Path to the Data.Yaml file
        epochs=cfg.EPOCHS, # Number of Epochs to Run
        patience=50,#if there is no improvement for "this" number of epochs early stopping will happen
        batch=cfg.BATCH_SIZE, #number of batch per image
        imgsz=cfg.IMG_SZ,#size of input images as integer
        save=True,#ave train checkpoints and predict results
        save_period=-1,#Save checkpoint every x epochs (disabled if < 1)
        cache=False,
        device=device,
        workers=8,#number of worker threads for data loading
        project=None,
        name='train18',
        exist_ok=False,
        pretrained=True,#(bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)
        optimizer=cfg.OPTIMIZER,#optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        # verbose=cfg.VERBOSE,#whether to print verbose output
        seed=cfg.SEED,#random seed
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,#resume training from last checkpoint
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        overlap_mask=True,
        mask_ratio=4,
        dropout=DROPOUT,#use dropout regularization
        val=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        source=None,
        show=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        stream_buffer=False,
        line_width=None,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        boxes=True,
        format='torchscript',
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=False,
        opset=None,
        workspace=4,
        nms=False,
        lr0=LEARNING_RATE,#initial learning rate
        lrf=0.01,#final learning rate
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,#label smoothing regularization
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        cfg=None,
        tracker='botsort.yaml',
        save_dir='runs/detect/train18'
)