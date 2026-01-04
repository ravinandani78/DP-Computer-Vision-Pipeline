import os
import yaml
import random
from shutil import copyfile
from loguru import logger


# Redudundant function keep in DpUtils
def read_class_labels(file_path):
    with open(file_path, 'r') as file:
        class_labels = file.read().splitlines()
    return class_labels

# please put Aug prefix in a config file
def check_if_aug(img_name):
    for aug_prefix in ["hflip_", "vflip_", "gblur_", "snp", "rscale_", "persp_", "rot_", "rrot90_"]:
        if aug_prefix in img_name:
            return True
    return False

# Set your input directory
def prepare_split_for_detection(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, CLASS_LABEL_FILE_PATH ,images_without_annotaiton, train_test_split_ratio, AUGMENTATION_TRAIN_ONLY = False ):


    train_percentage = train_test_split_ratio[0]
    test_percentage = train_test_split_ratio[1]
    val_percentage = train_test_split_ratio[2]


    # List all image files in the Images folder
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
    image_files = [f for f in image_files if f not in images_without_annotaiton]
    
    if AUGMENTATION_TRAIN_ONLY:
        Aug_files = [f for f in image_files if check_if_aug(f)]
    else:
        Aug_files = []
    
    image_files = [f for f in image_files if f not in Aug_files]
    
    # Shuffle the image files
    random.shuffle(image_files)

    # Calculate the number of images for each split
    num_train = int(len(image_files) * train_percentage)
    num_test = int(len(image_files) * test_percentage)
    num_val = len(image_files) - num_train - num_test

    # Create subdirectories for train, test, and val
    for split in ["train", "test", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    # Copy images and labels to the respective split folders
    for i, image_file in enumerate(image_files):
        src_image_path = os.path.join(IMAGES_DIR, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label_path = os.path.join(LABELS_DIR,  label_file)

        if i < num_train:
            dst_folder = "train"
        elif i < num_train + num_test:
            dst_folder = "test"
        else:
            dst_folder = "valid"

        dst_image_path = os.path.join(OUTPUT_DIR, dst_folder, "images", image_file)
        dst_label_path = os.path.join(OUTPUT_DIR, dst_folder, "labels", label_file)

        copyfile(src_image_path, dst_image_path)
        copyfile(src_label_path, dst_label_path)
        
      # Copy Aug Imgaes and labels to the respective split folders
    for i, image_file in enumerate(Aug_files):
        src_image_path = os.path.join(IMAGES_DIR, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label_path = os.path.join(LABELS_DIR,  label_file)

        #As all augmentation files should be in train
        dst_folder = "train"
        
        dst_image_path = os.path.join(OUTPUT_DIR, dst_folder, "images", image_file)
        dst_label_path = os.path.join(OUTPUT_DIR, dst_folder, "labels", label_file)

        copyfile(src_image_path, dst_image_path)
        copyfile(src_label_path, dst_label_path)

    # Create data.yaml file
    class_labels = read_class_labels(CLASS_LABEL_FILE_PATH)
    num_classes = len(class_labels)
    
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    
    data_yaml = {
    'train' : os.path.join(abs_output_dir, "train","images"),
    'val'   : os.path.join(abs_output_dir, "valid", "images"),
    'test'  : os.path.join(abs_output_dir, "test","images"),
    'nc'    : num_classes,
    'names' : class_labels
    }

    # Writing to a yaml file    
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file, default_flow_style=False, sort_keys =False),

    logger.success("Dataset split and data.yaml file created successfully")
