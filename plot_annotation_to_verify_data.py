#!/usr/bin/env python3
"""
Script to plot annotated labels on images for data verification.
This script reads images and their corresponding YOLO format label files,
draws bounding boxes with class labels, and saves the annotated images.

Usage:
    python plot_annotation_to_verify_data.py <data_folder_path>

The data folder should contain:
    - images/ folder with image files
    - labels/ folder with corresponding .txt label files
"""
# command to run the script: python3 plot_annotation_to_verify_data.py pipeline_output/processed_data/Detection/data/test
# command to run the script: python3 plot_annotation_to_verify_data.py pipeline_output/processed_data/Detection/data/test --output-dir plotted_images
import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import glob

class AnnotationVisualizer:
    def __init__(self, data_folder_path, output_dir="plotted_images"):
        """
        Initialize the annotation visualizer.
        
        Args:
            data_folder_path (str): Path to folder containing images/ and labels/ subdirectories
            output_dir (str): Directory to save plotted images
        """
        self.data_folder = Path(data_folder_path)
        self.images_dir = self.data_folder / "images"
        self.labels_dir = self.data_folder / "labels"
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Load class names (assuming classes.txt is in the data folder or parent)
        self.class_names = self._load_class_names()
        
        # Define colors for different classes
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (128, 128, 128), # Gray
            (0, 128, 0),    # Dark Green
        ]
        
    def _load_class_names(self):
        """Load class names from classes.txt file."""
        class_files = [
            self.data_folder / "classes.txt",
            self.data_folder.parent / "classes.txt",
            Path("raw_data/Data/classes.txt")
        ]
        
        for class_file in class_files:
            if class_file.exists():
                with open(class_file, 'r') as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(classes)} classes from {class_file}")
                return classes
        
        # Default class names if no file found
        print("Warning: No classes.txt found. Using default class names.")
        return [f"class_{i}" for i in range(10)]
    
    def _parse_yolo_annotation(self, label_file, img_width, img_height):
        """
        Parse YOLO format annotation file.
        
        Args:
            label_file (str): Path to label file
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            list: List of (class_id, x1, y1, x2, y2) tuples
        """
        annotations = []
        
        if not os.path.exists(label_file):
            return annotations
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                continue
                
            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to pixel coordinates
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                # Calculate bounding box coordinates
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                annotations.append((class_id, x1, y1, x2, y2))
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing line '{line}' in {label_file}: {e}")
                continue
                
        return annotations
    
    def _get_image_files(self):
        """Get all image files from the images directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(str(self.images_dir / ext)))
            image_files.extend(glob.glob(str(self.images_dir / ext.upper())))
            
        return sorted(image_files)
    
    def _get_corresponding_label_file(self, image_path):
        """Get the corresponding label file for an image."""
        image_name = Path(image_path).stem
        label_file = self.labels_dir / f"{image_name}.txt"
        return str(label_file)
    
    def plot_annotations_opencv(self, image_path, save_path):
        """
        Plot annotations on image using OpenCV and save the result.
        
        Args:
            image_path (str): Path to input image
            save_path (str): Path to save annotated image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False
            
        img_height, img_width = image.shape[:2]
        
        # Get corresponding label file
        label_file = self._get_corresponding_label_file(image_path)
        
        # Parse annotations
        annotations = self._parse_yolo_annotation(label_file, img_width, img_height)
        
        # Draw annotations
        for class_id, x1, y1, x2, y2 in annotations:
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Draw label background
            label_text = f"{class_name} ({class_id})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                image, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        # Save annotated image
        success = cv2.imwrite(save_path, image)
        if success:
            print(f"Saved annotated image: {save_path}")
            return True
        else:
            print(f"Error: Could not save image {save_path}")
            return False
    
    def process_all_images(self):
        """Process all images in the images directory."""
        image_files = self._get_image_files()
        
        if not image_files:
            print(f"No image files found in {self.images_dir}")
            return
            
        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {self.output_dir}")
        
        processed_count = 0
        for image_path in image_files:
            image_name = Path(image_path).name
            save_path = str(self.output_dir / f"annotated_{image_name}")
            
            if self.plot_annotations_opencv(image_path, save_path):
                processed_count += 1
                
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count}/{len(image_files)} images")
        print(f"Annotated images saved in: {self.output_dir}")
    
    def process_single_image(self, image_name):
        """Process a single image by name."""
        image_path = self.images_dir / image_name
        
        if not image_path.exists():
            print(f"Error: Image {image_path} not found")
            return False
            
        save_path = str(self.output_dir / f"annotated_{image_name}")
        return self.plot_annotations_opencv(str(image_path), save_path)


def main():
    parser = argparse.ArgumentParser(description='Plot annotations on images for data verification')
    parser.add_argument('data_folder', help='Path to folder containing images/ and labels/ subdirectories')
    parser.add_argument('--output-dir', default='plotted_images', help='Output directory for annotated images')
    parser.add_argument('--single-image', help='Process only a single image (provide image filename)')
    
    args = parser.parse_args()
    
    # Validate input folder
    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        print(f"Error: Data folder {data_folder} does not exist")
        sys.exit(1)
        
    images_dir = data_folder / "images"
    labels_dir = data_folder / "labels"
    
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        sys.exit(1)
        
    if not labels_dir.exists():
        print(f"Error: Labels directory {labels_dir} does not exist")
        sys.exit(1)
    
    # Create visualizer
    visualizer = AnnotationVisualizer(args.data_folder, args.output_dir)
    
    # Process images
    if args.single_image:
        visualizer.process_single_image(args.single_image)
    else:
        visualizer.process_all_images()


if __name__ == "__main__":
    main()
