"""
Trainer Module assigns the task to the indiviual training pipelines.
"""

import sys
sys.path.append('Modules')
from loguru import logger
from .Classification import Classifier
from .Detection import Detector

class Trainer():
    def __init__(self, cfg):
        logger.info("Training Initialized\n")
        self.cfg = cfg
        logger.info("Initializing with Configuration ::")
        for attr in dir(self.cfg):
            if not attr.startswith("__"):
                logger.opt(ansi=True).info(f"<blue>{attr:<15}</blue> <red>{getattr(self.cfg, attr)}</red>")
        
    #This function is private hence cannot be called outside the class, only internal calls are allowed.
    # as two underscores at start of the function name.
    @property
    def __func_map(self):

        return {
            'classification': Classifier,
            'detection'     : Detector,
            'segmentation'  : NotImplementedError,
            'clustering'    : NotImplementedError
        }
        
        
    def __call__(self):
        func = self.__func_map[self.cfg.TASK]
        return func(self.cfg)
    
        