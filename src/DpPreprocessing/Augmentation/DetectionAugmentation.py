"""
    This File Definies all the Augmentations for Detection tasks. The Base Logic is written in base class
    called BaseAugmentation.
"""
# Reference From https://github.com/albumentations-team/albumentations
import albumentations as A
from .BaseAugmentation import Augment
from .CustomAugmentations import CustomGaussianBlur, CustomSaltandPepper
import argparse


def locals2config(lcls):
    """
    this function takes any class inputs and convert into a config object
    """
    lcls.pop('self')
    return argparse.Namespace(**lcls)
    
# -----------------------------------------------Resizing Augmentation---------------------------------------------------#
class Resize(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, RESIZE_DIM):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = ""
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.Resize(p = 1, height = RESIZE_DIM[0], width = RESIZE_DIM[1])
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )
        
    def __call__(self):
            self.pipeline(self.cnfg, self.transform)
            
            
# ------------------------------------------HorizontalFlip Augmentation--------------------------------------------------#
class HorizontalFlip(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "hflip_" if not c else f"hfilp_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.HorizontalFlip(p = 1)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )
        
    def __call__(self):
            self.pipeline(self.cnfg, self.transform)
            
            
# ------------------------------------------VerticalFlip Augmentation--------------------------------------------------#
class VerticalFlip(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "vflip_" if not c else f"vfilp_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.VerticalFlip(p = 1)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
        

# ------------------------------------------GaussianBlur Augmentation--------------------------------------------------#
class GaussianBlur(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "gblur_" if not c else f"gblur_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [   
                    CustomGaussianBlur(p=1)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
        
        
# ------------------------------------------SaltandPepper Augmentation--------------------------------------------------#        
class SaltandPepper(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "snp_" if not c else f"snp_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [   
                    CustomSaltandPepper(p=1, intensity = 0.10)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
        
        
# -------------------------------------------RandomScale Augmentation--------------------------------------------------#
class RandomScale(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "rscale_" if not c else f"rscale_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.RandomScale(p = 1)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
  

# -------------------------------------------Perspective Augmentation--------------------------------------------------#
class Perspective(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "persp_" if not c else f"persp_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.Perspective(p = 1)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
              

# -------------------------------------------Rotate Augmentation--------------------------------------------------#
class Rotate(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "rot_" if not c else f"rot_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.Rotate(p = 1)
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
        
# -------------------------------------------RandomRotate90 Augmentation-------------------------------------------#
class RandomRotate90(Augment):
    def __init__(self, IMAGE_LIST, IMAGES_DIR, LABELS_DIR, OUTPUT_IMAGE_DIR, OUTPUT_LABELS_DIR, c = None):
        self.cnfg = locals2config(locals())
        self.cnfg.PREFIX = "rrot90_" if not c else f"rrot90_{c}_" 
        self.cnfg.SUFFIX = ""
        self.transform = A.Compose(
                [
                    A.RandomRotate90(p = 1)
                    
                    
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]),
            )

    def __call__(self):
        self.pipeline(self.cnfg, self.transform)
        



