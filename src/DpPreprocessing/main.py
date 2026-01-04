import os
import shutil
from loguru import logger
# from DpPreprocessing.Preparation.fetchData import download_from_s3
from DpPreprocessing.FormatConversion.custom_conversion import convert2yolo
from DpPreprocessing.Augmentation import DetectionAugmentation, FileSelection
from DpPreprocessing.Preparation.Detection2Classification import crop_and_save_images
from DpPreprocessing.Split.classification import prepare_split_for_classification
from DpPreprocessing.Split.detection import prepare_split_for_detection
from DpUtils.DataUtils import create_file_distributions, find_images_without_annotations
from accessify import private
import yaml

class PrepareDataset(object):
    def __init__(self, config) -> None:
        logger.opt(ansi=True).info("<E>Pipeline :: PrepareDataset initialized.</E>")
        self.config = config
        self.data_config = config['data']
        self.prep_config = config['preprocessing']

        # Define paths from config
        self.raw_data_dir = self.data_config['raw_data_dir']
        self.processed_dir = self.data_config['processed_dir']
        
        self.images_dir = os.path.join(self.raw_data_dir, self.data_config['images_dir'])
        self.labels_dir = os.path.join(self.raw_data_dir, self.data_config['labels_dir'])
        self.annotation_dir = os.path.join(self.raw_data_dir, self.data_config['annotation_dir'])
        self.class_label_file = os.path.join(self.raw_data_dir, self.data_config['class_label_file'])

        self.det_aug_img_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['augmented']['images'])
        self.det_aug_label_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['augmented']['labels'])
        self.det_orig_img_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['original_sized']['images'])
        self.det_orig_label_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['original_sized']['labels'])
        self.det_resized_img_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['resized']['images'])
        self.det_resized_label_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['resized']['labels'])
        self.det_split_path = os.path.join(self.processed_dir, self.prep_config['output']['detection']['split_path'])
        
        self.class_orig_path = os.path.join(self.processed_dir, self.prep_config['output']['classification']['original'])
        self.class_main_csv = os.path.join(self.processed_dir, self.prep_config['output']['classification']['main_csv'])
        self.class_train_csv = os.path.join(self.processed_dir, self.prep_config['output']['classification']['train_csv'])
        self.class_test_csv = os.path.join(self.processed_dir, self.prep_config['output']['classification']['test_csv'])

        self.dirs_to_create = [
            self.labels_dir,
            self.det_aug_img_path,
            self.det_aug_label_path,
            self.det_orig_img_path,
            self.det_orig_label_path,
            self.det_resized_img_path,
            self.det_resized_label_path,
            self.det_split_path,
            self.class_orig_path,
        ]

    def __call__(self):
        self.remove_previous_data()
        self.create_directories()
        self.download_data(client=None)
        self.create_labels()
        self.images_without_annotation = self.find_images_without_annotation()
        self.augmentation_file_selection()
        self.apply_augmentation()
        self.copy_to_respective_folders()
        self.resize_images()
        self.crop_for_classification()
        self.classification_split()
        self.detection_split()

    @private
    def remove_previous_data(self):
        logger.info("Removing Previous Processed Data")
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)

    @private
    def create_directories(self):
        logger.info("Creating fresh Directories for Final Sets")
        for dir_ in self.dirs_to_create:
            os.makedirs(dir_, exist_ok=True)

    @private
    def download_data(self, client, *args, **kwargs):
        logger.info(f"Downloading Dataset using {client} client")
        # if client == 's3':
        #     download_from_s3(*args, **kwargs)

    @private
    def create_labels(self):
        logger.info("Converting Annotation JSON to Label txt files")
        os.makedirs(self.labels_dir, exist_ok=True)
        for annotation_file in os.listdir(self.annotation_dir):
            convert2yolo(
                os.path.join(self.annotation_dir, annotation_file),
                self.images_dir,
                self.labels_dir,
                self.class_label_file
            )

    @private
    def find_images_without_annotation(self):
        os.makedirs(self.labels_dir, exist_ok=True)
        return find_images_without_annotations(self.images_dir, self.labels_dir)

    @private
    def augmentation_file_selection(self):
        aug_selection_config = self.prep_config['augmentation']['file_selection']
        logger.info(f"Augmentation Method :: {aug_selection_config['method']}")
        
        selection = FileSelection.Selection(self.images_dir, self.labels_dir, **aug_selection_config)
        images_for_augmentation = selection()
        images_for_augmentation = [f for f in images_for_augmentation if f not in self.images_without_annotation]
        self.sublists = create_file_distributions(self.images_dir, self.prep_config['augmentation']['dist'], images_for_augmentation)

    @private
    def apply_augmentation(self):
        logger.info("Applying Augmentations")
        aug_config = self.prep_config['augmentation']
        cycle = aug_config['cycle']
        exp = aug_config['exp']
        
        common_args = (self.images_dir, self.labels_dir, self.det_aug_img_path, self.det_aug_label_path)

        if exp:
            logger.info(f"Applying Augmentations in Exponential running for {cycle} cycles on all augmentation images")
            all_imgs = [im for val in self.sublists.values() for im in val]
            for i in range(cycle):
                for aug_name in self.sublists.keys():
                    getattr(DetectionAugmentation, aug_name)(all_imgs, *common_args, c=i + 1)()
        else:
            for aug_name, img_list in self.sublists.items():
                getattr(DetectionAugmentation, aug_name)(img_list, *common_args)()

    @private
    def copy_to_respective_folders(self):
        logger.info("Copying Datasets to their respective paths")
        for f in os.listdir(self.det_aug_img_path):
            shutil.copy2(os.path.join(self.det_aug_img_path, f), self.det_orig_img_path)
        for f in os.listdir(self.det_aug_label_path):
            shutil.copy2(os.path.join(self.det_aug_label_path, f), self.det_orig_label_path)

        for image in os.listdir(self.images_dir):
            shutil.copy(os.path.join(self.images_dir, image), os.path.join(self.det_orig_img_path, image))
        for label in os.listdir(self.labels_dir):
            shutil.copy(os.path.join(self.labels_dir, label), os.path.join(self.det_orig_label_path, label))

    @private
    def resize_images(self):
        logger.info("Resizing the Images")
        DetectionAugmentation.Resize(
            os.listdir(self.det_orig_img_path),
            self.det_orig_img_path,
            self.det_orig_label_path,
            self.det_resized_img_path,
            self.det_resized_label_path,
            tuple(self.prep_config['resize_dim'])
        )()

    @private
    def crop_for_classification(self):
        logger.info("Cropping Images for classification Dataset")
        crop_and_save_images(
            self.det_orig_img_path,
            self.det_orig_label_path,
            self.class_label_file,
            self.class_orig_path,
            self.class_main_csv
        )

    @private
    def classification_split(self):
        logger.info("Preparing Classification split")
        classification_data_yaml_path = prepare_split_for_classification(
            self.class_orig_path,
            self.class_label_file,
            tuple(self.prep_config['split']['classification_ratio'])
        )
        # Update the config with the new path
        self.config['training']['classification']['data_path'] = classification_data_yaml_path
        with open('configs/config.yaml', 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)

    @private
    def detection_split(self):
        logger.info("Preparing Detection split")
        prepare_split_for_detection(
            self.det_resized_img_path,
            self.det_resized_label_path,
            self.det_split_path,
            self.class_label_file,
            self.images_without_annotation,
            tuple(self.prep_config['split']['detection_ratio']),
            AUGMENTATION_TRAIN_ONLY=self.prep_config['augmentation']['train_only']
        )