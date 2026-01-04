import os
import random
import pandas as pd 
from loguru import logger

class Selection(object):
    def __init__(
                self, images_path : str = None, labels_path: str = None ,
                method : str = 'manual', prob : float=None, csv_path : str = None
                ):
        self.images_path = images_path
        self.labels_path = labels_path
        self.method = method
        self.prob = prob
        self.csv_path = csv_path
        self.image_names = os.listdir(self.images_path)
        
    def image2label(self, image_name):
        return image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        
    
    def manual_selection(self):
        assert self.csv_path is not None , self.raise_assertion['manual']
        images_for_augmentation = pd.read_csv(self.csv_path)['filenames']
        # print(images_for_augmentation)
        # labels_for_augmentation = [self.image2label(image_name) for image_name in images_for_augmentation ]
        
        images_not_for_augmentation = [image for image in self.image_names if image not in images_for_augmentation ]
        # labels_not_for_augmentation = [self.image2label(image_name) for image_name in images_not_for_augmentation ]
        return images_for_augmentation
        
        
    def auto_selection(self):
        raise NotImplementedError
    
    
    def random_selection(self):
        assert self.prob is not None , self.raise_assertion['random']
        random.shuffle(self.image_names)
        aug_size = int(len(self.image_names) * self.prob)
        images_for_augmentation = self.image_names[:aug_size]
        # labels_for_augmentation = [self.image2label(image_name) for image_name in images_for_augmentation ]
        # images_not_for_augmentation = self.image_names[aug_size:]
        # labels_not_for_augmentation = [self.image2label(image_name) for image_name in images_not_for_augmentation ]
        return images_for_augmentation
        
         
    #This function is private hence cannot be called outside the class, only internal calls are allowed.
    # as two underscores at start of the function name.
    @property
    def __func_map(self):
        return {
            'manual' : self.manual_selection,
            'auto'   : self.auto_selection,
            'random' : self.random_selection
        }
        
        
    def __call__(self):
        func = self.__func_map[self.method]
        return func()
    
    
    @property
    def raise_assertion(self):
        return {
            'manual' : """
                    Manual Selection Requires csv_path for list of images for augmentation.
                    
                    selection = Selection(images_path, labels_path, method = "manual", csv_path = 'images_for_augmentation.csv' )
                    
                    The csv file should have a column named "filenames" which contains the name
                    of the images for Augmentation. Please keep only the file names with extension
                    full path is not required.
                    
                    For Example : 

                    filenames
                    122hk1h2kh1h2j2.jpg
                    1232908ujhkslkj.jpg
                    ...
                    ..
                    .
                    
            """,
            'random':"""
                    Random Selection Requires a probability for randomly selecting the images for augmentation.
                    
                    selection = Selection(images_path, labels_path, method = "manual", prob = 0.20 )
                    
                    For Example if there are 1000 images and if probability given is .20 it will select 200 random images 
                    for Augmentation
            """
        
    }
        