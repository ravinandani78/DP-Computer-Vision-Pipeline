
import os
import shutil
import yaml
from collections import defaultdict
from sklearn.model_selection import train_test_split
from loguru import logger

def prepare_split_for_classification(source_dir, class_label_file, split_ratio):
    """
    Organizes classification data into train, validation, and test sets.
    """
    base_target_dir = os.path.join(os.path.dirname(source_dir), "Split")

    if os.path.exists(base_target_dir):
        logger.info(f"Removing existing directory: {base_target_dir}")
        shutil.rmtree(base_target_dir)

    logger.info(f"Creating new directory: {base_target_dir}")
    os.makedirs(base_target_dir)

    train_dir = os.path.join(base_target_dir, 'train')
    val_dir = os.path.join(base_target_dir, 'val')
    test_dir = os.path.join(base_target_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files_by_class = defaultdict(list)
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            try:
                class_id = filename.rsplit('.', 2)[1]
                files_by_class[class_id].append(filename)
            except IndexError:
                logger.warning(f"Could not parse class ID from filename: {filename}")

    for class_id, files in files_by_class.items():
        if len(files) < 3:
            logger.warning(f"Class {class_id} has fewer than 3 images, cannot split. Skipping.")
            continue

        os.makedirs(os.path.join(train_dir, class_id), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_id), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_id), exist_ok=True)

        train_val_files, test_files = train_test_split(files, test_size=split_ratio[2], random_state=42)
        train_files, val_files = train_test_split(train_val_files, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=42)

        for filename in train_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, class_id, filename))
        for filename in val_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, class_id, filename))
        for filename in test_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(test_dir, class_id, filename))

    logger.info("Classification data organization complete.")

    # Create data.yaml for classification
    with open(class_label_file, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
    
    num_classes = len(class_labels)
    
    abs_output_dir = os.path.abspath(base_target_dir)

    data_yaml = {
        'train': os.path.join(abs_output_dir, "train"),
        'val': os.path.join(abs_output_dir, "val"),
        'test': os.path.join(abs_output_dir, "test"),
        'nc': num_classes,
        'names': class_labels
    }

    data_yaml_path = os.path.join(base_target_dir, "data.yaml")
    with open(data_yaml_path, "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file, default_flow_style=False, sort_keys=False)
    
    logger.success(f"Classification data.yaml file created successfully at {data_yaml_path}")

    return data_yaml_path
