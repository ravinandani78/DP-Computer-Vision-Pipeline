
def Detector(cfg):
    if cfg.MODELTYPE == 'yolov5':
        from Modules.DpTraining.Detection.yolov5.DpTrain import Train
    elif cfg.MODELTYPE == 'yolov8':
        from Modules.DpTraining.Detection.yolov8.DpTrain import Train
    else :
        raise NotImplementedError
    return Train(cfg)