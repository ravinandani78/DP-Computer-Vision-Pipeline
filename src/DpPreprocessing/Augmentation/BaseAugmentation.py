from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm 


class Augment(ABC):    
    """
    Base class for Augmentation , All logic will be written here and this gets inherited in indivual augmentation.
    """
    def __init__(self, *args, **kwargs):
        pass

    def pipeline(self, cnfg, transform):
        count = 0
        for image_name in tqdm(cnfg.IMAGE_LIST, leave = False, desc = f"Applying Augmentation   | {str(self.__class__.__name__)}"):
            try:
                base_name = os.path.splitext(os.path.basename(image_name))[0]
                image_ext = os.path.splitext(os.path.basename(image_name))[1]
                label_ext = '.txt'
                self.cnfg.IMAGE_FILE_PATH = os.path.join(cnfg.IMAGES_DIR, base_name + image_ext)
                self.cnfg.LABEL_FILE_PATH = os.path.join(cnfg.LABELS_DIR, base_name + label_ext)
                
                image, bboxes, category_ids = self.preprocess(cnfg.IMAGE_FILE_PATH, cnfg.LABEL_FILE_PATH)
                aug_image, aug_bboxes = self.apply_transformation(transform, image, bboxes, category_ids)
                
                self.saveimage(aug_image, cnfg.OUTPUT_IMAGE_DIR, base_name, image_ext, cnfg.PREFIX, cnfg.SUFFIX)
                self.savelabel(aug_bboxes, cnfg.OUTPUT_LABELS_DIR, cnfg.PREFIX, cnfg.SUFFIX)
                count+=1
            except:
                print(image_name) 
                print(count)
                raise Exception

    def preprocess(self, image_file_path, label_file_path):
        """
        This Function is to Preprocess the image and label file for Augmenation

        Args:
            image_file_path (str): Path of the image for Augmentation.
            label_file_path (str): Path of the Label for Augmentation.

        Returns:
            tuple : image, updated_bboxes, category_ids
        """
        
        #Creating a basename so that can be used for label names (img:123.jpg, base: 123, label :base+.txt 123.txt)
        self.base_name = os.path.splitext(os.path.basename(image_file_path))[0]
        
        try:
            df_text=pd.read_table(label_file_path,header=None,sep=' ')
        except: # This is patch to handle empty label files a.k.a images with no labels
            image = Image.open(image_file_path)
            image = np.array(image)  
            updated_bboxes = []
            category_ids = []
            return image, updated_bboxes, category_ids
        
        # label_array = np.array(df_text)
        label_array = np.array(df_text)
        bboxes = label_array[:, 1:5]
        category_ids = label_array[:, 0].astype(int)
        image = Image.open(image_file_path)
        image = np.array(image)        
        updated_bboxes = self.bbox_correction(image, bboxes)
        return image, updated_bboxes, category_ids


    def bbox_correction(self, image, bboxes):
        updated_bboxes = bboxes.copy()
        for i, bbox in enumerate(updated_bboxes):
            x, y, h, w = bbox
            x1 = x - w / 2
            x2 = x1 + w
            y1 = y - h / 2
            y2 = y1 + h

            if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1:
                updated_bboxes[i][2] = np.abs(updated_bboxes[i][2] - 0.5/ image.shape[0])
                updated_bboxes[i][3] = np.abs(updated_bboxes[i][3] - 0.5/ image.shape[1])
        return updated_bboxes

    def apply_transformation(self, transform, image, bboxes, category_ids):
        if len(bboxes) == 0:
            return image, []
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        aug_image = transformed["image"]
        aug_bboxes = np.column_stack(
            (np.array(transformed["category_ids"]).astype(int), transformed["bboxes"])
        )
        return aug_image, aug_bboxes

    def category2names(self, category_ids, LABEL_MAPS=None):
        return {int(c): str(c) for c in category_ids}

    def saveimage(self, image, OUTPUT_IMAGE_DIR, base_name, image_ext, prefix, suffix):
        filename = os.path.join(
            OUTPUT_IMAGE_DIR, prefix + self.base_name + suffix + image_ext)
        self.convert2pil(image).save(filename)

    def savelabel(self, bboxes, OUTPUT_LABELS_DIR, prefix, suffix):
        filename = os.path.join(
            OUTPUT_LABELS_DIR, prefix + self.base_name + suffix + ".txt"
        )
        pd.DataFrame(bboxes).to_csv(filename, sep = " ", index = False, header = False)
        # np.savetxt(filename, bboxes, fmt="%.6f")

    def convert2pil(self, arr):
        return Image.fromarray(arr)

    def identify_annotation_type(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    def __repr__(self):
        pass
