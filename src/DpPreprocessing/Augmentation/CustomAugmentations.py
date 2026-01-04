from albumentations.augmentations.utils import _maybe_process_in_chunks, preserve_shape
from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug.augmenters as iaa
import numpy as np
import cv2

def apply_snp(img, intensity ):
    aug = iaa.SaltAndPepper(intensity,per_channel=True)
    return aug(images = [img])[0]

@preserve_shape
def salt_and_pepper(img: np.ndarray, intensity:float) -> np.ndarray:
    blur_fn = _maybe_process_in_chunks(apply_snp, intensity = intensity)
    return blur_fn(img)

class CustomSaltandPepper(ImageOnlyTransform):
    def __init__(
        self,
        intensity = .20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.intensity = intensity

 
    def apply(self, img: np.ndarray ,**params) -> np.ndarray:
        return salt_and_pepper(img, self.intensity)
    
    
    
def apply_blur(img):
    width = img.shape[0]
    temp_kernel = int(width*.005)
    if temp_kernel%2==0:
        kernel = (temp_kernel+1, temp_kernel+1)
    else:
        kernel = (temp_kernel, temp_kernel)

    return cv2.GaussianBlur(img, kernel, 0)

@preserve_shape
def gaussian_blur(img: np.ndarray) -> np.ndarray:
    blur_fn = _maybe_process_in_chunks(apply_blur)
    return blur_fn(img)

class CustomGaussianBlur(ImageOnlyTransform):
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        
    def apply(self, img: np.ndarray ,**params) -> np.ndarray:
        return gaussian_blur(img)
    