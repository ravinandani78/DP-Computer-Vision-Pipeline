import random
import os
from loguru import logger

def find_images_without_annotations(IMAGES_DIR, LABELS_DIR):
    IMAGE_FILES = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    LABEL_FILES = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
    return [img for img in IMAGE_FILES if f"{os.path.splitext(img)[0]}.txt" not in LABEL_FILES]

    
def create_file_distributions(file_dir, dist_dict, images_with_annotaiton):
    # Calculate the number of images for each category
    image_list  =os.listdir(file_dir)
    image_list = [f for f in image_list if f in images_with_annotaiton ]
    num_images = len(image_list)
    sublist_lengths = {key: int(value * num_images) for key, value in dist_dict.items()}
    # Shuffle the input image list randomly
    random.shuffle(image_list)

    # Create sublists based on the percentage values
    sublists = {}
    start_idx = 0

    for category, length in sublist_lengths.items():
        end_idx = start_idx + length
        sublists[category] = image_list[start_idx:end_idx]
        start_idx = end_idx

    # Any remaining images not included in the specified percentages will be added to one of the sublists
    for idx in range(start_idx, num_images):
        category = random.choice(list(sublists.keys()))
        sublists[category].append(image_list[idx])

    # Print or use sublists as needed
    
    [logger.info(f'Augmentation Split  :: {category: <15} - {len(sublist)}') for category, sublist in sublists.items()]
    
    return sublists
