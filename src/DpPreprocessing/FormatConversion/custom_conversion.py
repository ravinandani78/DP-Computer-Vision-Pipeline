from loguru import logger
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations, convert_bboxes_from_albumentations
from glob import glob
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

def check_bbox(bbox) -> None:
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
            # raise ValueError(f"Expected {name} for bbox {bbox} to be in the range [0.0, 1.0], got {value}.")
            return False
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        # raise ValueError(f"x_max is less than or equal to x_min for bbox {bbox}.")
        return False
    if y_max <= y_min:
        return False
        # raise ValueError(f"y_max is less than or equal to y_min for bbox {bbox}.")
    return True


def validate_box_coco(x_min, y_min, width, height, rows, cols):
    x_max = x_min + width
    y_max = y_min + height
    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows
    bbox = (x_min, y_min, x_max, y_max)
    return check_bbox(bbox)

@logger.catch()
def convert2yolo(ANNOTATION_FILE, IMAGES_DIR, LABELS_DIR, output_file):
    with open(ANNOTATION_FILE, 'r') as f:
        JSON = json.load(f)
        
    # Write CLASS FILE
    categories = JSON['categories']
    max_categories = max([int(category['id']) for category in categories])
    temp_classes = ['unknown' for i  in range(max_categories)]
    
    for category in categories:
        # print(int(category['id']))
        temp_classes[int(category['id'])-1] = category['name']
    
    updated_classes = temp_classes
    save_class_names_to_file(updated_classes, output_file)

    blank_count= 0
    for i in range(len(JSON['images'])):
        # try:
        file_name = JSON['images'][i]['file_name']    
        image_id  = JSON['images'][i]['id']
        img_width = JSON['images'][i]['width']
        img_height = JSON['images'][i]['height']
        image_file = os.path.join(IMAGES_DIR, file_name)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        image_ext = os.path.splitext(os.path.basename(image_file))[1]
        rows = img_height
        cols = img_width
        label_ext = '.txt'
        LABEL_FILE_PATH = os.path.join(LABELS_DIR, base_name + label_ext)    

        ann_data = [JSON['annotations'][i] for i in range(len(JSON['annotations'])) if JSON['annotations'][i]['image_id'] == image_id]
        
        if len(ann_data) == 0:
            blank_count +=1
            continue
        ann = np.array([[ann_box['category_id']-1] + ann_box['bbox'] for ann_box in ann_data])
        category_ids = ann[:, 0].astype(np.float64)
        coco_bboxes = ann[:, 1:5]
        coco_bboxes_copy = copy.deepcopy(coco_bboxes)
        validations = np.array([validate_box_coco(xi[0], xi[1], xi[2], xi[3], rows, cols) for xi in coco_bboxes_copy])
        coco_bboxes = coco_bboxes[validations]

        album_bboxes = convert_bboxes_to_albumentations(
            coco_bboxes, 'coco', rows, cols, check_validity=True
        ) 

        yolo_bboxes = convert_bboxes_from_albumentations(
            album_bboxes, 'yolo', rows, cols, True
        )

        yolo_boxes_with_cats = [[category_ids[i]]+list(v) for i, v in enumerate(yolo_bboxes)]

        pd.DataFrame(yolo_boxes_with_cats).to_csv(LABEL_FILE_PATH, sep = " ", index = False, header = False)

   
    logger.success(f"Out of {len(JSON['images'])} Images {len(JSON['images'])- blank_count} images were processed")            
    logger.critical(f"Out of {len(JSON['images'])} Images {blank_count} images were Failed")            
    make_shell_labels(IMAGES_DIR, LABELS_DIR)


def image2label(image_name):
    return image_name.replace('.jpg', '.txt').replace('.png', '.txt')


def make_shell_labels(IMAGES_DIR, LABELS_DIR):
    for image_name in os.listdir(IMAGES_DIR):
        label_name = image2label(image_name)
        labe_file_path = os.path.join(LABELS_DIR, label_name)
        if not os.path.exists(labe_file_path):
            with open(labe_file_path, 'w') as f:pass
                



def save_class_names_to_file(class_names, output_file):
    with open(output_file, 'w') as file:
        for class_name in class_names:
            file.write(f'{class_name}\n')


    
    