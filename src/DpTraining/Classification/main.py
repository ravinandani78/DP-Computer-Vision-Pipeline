
def Classifier(cfg):
    if cfg.MODELTYPE == 'fastai':
        from .fastai.DpTrain import Train
    else :
        raise NotImplementedError
    return Train(cfg)