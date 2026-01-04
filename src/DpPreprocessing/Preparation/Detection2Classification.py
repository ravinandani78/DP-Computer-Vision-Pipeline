import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm 


# Function to read class labels from a file
def read_class_labels(file_path):
    with open(file_path, 'r') as file:
        class_labels = file.read().splitlines()
    return class_labels

# Function to read labels from a file
def read_labels(file_path):
    try:
        df_text=pd.read_table(file_path,header=None, delim_whitespace=True)
    except:
        df_text = pd.DataFrame()
    labels = np.array(df_text.copy())
    return labels

# Function to crop and save images
def crop_and_save_images(images_folder, labels_folder, class_labels_file, output_folder, main_file_path):
    class_labels = read_class_labels(class_labels_file)
    data_list = []
    for image_file in tqdm(os.listdir(images_folder), leave = False) :
        image_path = os.path.join(images_folder, image_file)
        labels_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt'))

        if os.path.exists(labels_path):
            img = cv2.imread(image_path)
            height, width, _ = img.shape

            labels = read_labels(labels_path)
            # cropped_image = img.copy()
            for idx, label in enumerate(labels):
                class_id, x_center, y_center, width, height = label
                x_min = int((x_center - width/2) * img.shape[1])
                y_min = int((y_center - height/2) * img.shape[0])
                x_max = int((x_center + width/2) * img.shape[1])
                y_max = int((y_center + height/2) * img.shape[0])

                # Crop the image based on the bounding boxes
                cropped_image = img[y_min:y_max, x_min:x_max]
                class_label = class_labels[int(class_id)]

                output_img_name = "crp_"+ image_file.split(".jpg")[0] + "--"+str(idx)+"_"+str(class_id)+".jpg"
                data_list.append({'image_name':output_img_name, 'label' : class_label})
                cv2.imwrite(os.path.join(output_folder,output_img_name), cropped_image)
    
    pd.DataFrame(data_list).to_csv(os.path.join(output_folder, '..', 'main.csv'), sep =',', index = False)

if __name__ == '__main__':    
    # Example usage
    images_folder = 'path/to/images'
    labels_folder = 'path/to/labels'
    class_labels_file = 'path/to/class_labels.txt'
    output_folder = 'path/to/output_folder'

    crop_and_save_images(images_folder, labels_folder, class_labels_file, output_folder)
