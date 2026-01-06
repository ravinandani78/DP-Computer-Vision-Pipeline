"""
Streamlit Demo for DP Data Preparation Pipeline for Computer Vision
A comprehensive web interface to manage and run the entire ML pipeline.
"""

import streamlit as st
import yaml
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import json
from typing import Dict, Any
import time
import sys  
# Page configuration
st.set_page_config(
    page_title="DP Computer Vision Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2ca02c;
        margin: 1.5rem 0 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load the current configuration from config.yaml"""
    config_path = "configs/config.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        return None

def save_temp_config(config: Dict[Any, Any]) -> str:
    """Save configuration to a temporary file and return the path"""
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, "temp_config.yaml")
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return temp_config_path

def run_pipeline_command(command: str, module_name: str):
    """Run a pipeline command and display the output"""
    with st.spinner(f"Running {module_name}..."):
        try:
            # Create a placeholder for real-time output
            output_placeholder = st.empty()
            
            # Use sys.executable to ensure same Python interpreter
            # Replace 'python3' with sys.executable in the command
            if command.startswith('python3'):
                command = command.replace('python3', sys.executable, 1)
            
            # Run the command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.getcwd(),
                env=os.environ.copy()  # Ensure environment variables are passed
            )
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
                    # Show last 10 lines of output
                    recent_output = '\n'.join(output_lines[-10:])
                    output_placeholder.code(recent_output)
            
            rc = process.poll()
            
            if rc == 0:
                st.markdown(f'<div class="success-box">✅ {module_name} completed successfully!</div>', 
                           unsafe_allow_html=True)
                
                # Show full output in expander
                with st.expander("View Full Output"):
                    st.code('\n'.join(output_lines))
            else:
                st.markdown(f'<div class="error-box">❌ {module_name} failed with exit code {rc}</div>', 
                           unsafe_allow_html=True)
                st.code('\n'.join(output_lines))
                
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error running {module_name}: {str(e)}</div>', 
                       unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 DP Computer Vision Pipeline Demo</h1>', 
                unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Pipeline Modules")
    selected_module = st.sidebar.radio(
        "Select Module:",
        ["📊 Overview", "🔄 Data Preprocessing", "🚀 Model Training", "📈 Model Evaluation"]
    )
    
    # Overview Section
    if selected_module == "📊 Overview":
        st.markdown('<div class="module-header">📊 Pipeline Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔄 Data Preprocessing")
            st.info("""
            - Format conversion (JSON to YOLO)
            - Data augmentation
            - Image resizing
            - Dataset splitting
            - Classification crop generation
            """)
        
        with col2:
            st.markdown("### 🚀 Model Training")
            st.info("""
            - YOLOv8 Object Detection
            - YOLOv8 Classification
            - Hyperparameter tuning
            - MLflow experiment tracking
            - Model checkpointing
            """)
        
        with col3:
            st.markdown("### 📈 Model Evaluation")
            st.info("""
            - Comprehensive metrics
            - Confusion matrices
            - Performance visualizations
            - Automated reporting
            - MLflow integration
            """)
        
        st.markdown('<div class="info-box">💡 Use the sidebar to navigate between different pipeline modules. Each module allows you to configure parameters and run the respective pipeline stage.</div>', 
                   unsafe_allow_html=True)
    
    # Data Preprocessing Module
    elif selected_module == "🔄 Data Preprocessing":
        st.markdown('<div class="module-header">🔄 Data Preprocessing Configuration</div>', 
                   unsafe_allow_html=True)
        
        # Create a copy of config for modifications
        preprocessing_config = config.copy()
        
        # Dataset Selection
        st.markdown('<div class="sub-header">📁 Dataset Configuration</div>', unsafe_allow_html=True)
        
        use_default_dataset = st.checkbox("Use Default Dataset", value=True)
        
        if not use_default_dataset:
            st.markdown("**Upload Custom Dataset**")
            uploaded_file = st.file_uploader(
                "Upload a ZIP file containing 'images' and 'Annotation_Json_File' folders",
                type=['zip']
            )
            
            if uploaded_file is not None:
                # Handle file upload (implementation would extract and set paths)
                st.success("Dataset uploaded successfully!")
                st.info("Custom dataset upload functionality would be implemented here.")
        
        # Data Paths Configuration
        st.markdown('<div class="sub-header">📂 Data Paths</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            raw_data_dir = st.text_input(
                "Raw Data Directory",
                value=preprocessing_config['data']['raw_data_dir']
            )
            images_dir = st.text_input(
                "Images Directory",
                value=preprocessing_config['data']['images_dir']
            )
        
        with col2:
            processed_dir = st.text_input(
                "Processed Data Directory",
                value=preprocessing_config['data']['processed_dir']
            )
            labels_dir = st.text_input(
                "Labels Directory",
                value=preprocessing_config['data']['labels_dir']
            )
        
        # Image Processing Configuration
        st.markdown('<div class="sub-header">🖼️ Image Processing</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            resize_width = st.number_input(
                "Image Width",
                min_value=64,
                max_value=2048,
                value=preprocessing_config['preprocessing']['resize_dim'][0],
                step=32
            )
        
        with col2:
            resize_height = st.number_input(
                "Image Height",
                min_value=64,
                max_value=2048,
                value=preprocessing_config['preprocessing']['resize_dim'][1],
                step=32
            )
        
        # Augmentation Configuration
        st.markdown('<div class="sub-header">🔄 Data Augmentation</div>', unsafe_allow_html=True)
        
        aug_config = preprocessing_config['preprocessing']['augmentation']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            horizontal_flip = st.slider(
                "Horizontal Flip",
                min_value=0.0,
                max_value=1.0,
                value=aug_config['dist']['HorizontalFlip'],
                step=0.05
            )
        
        with col2:
            vertical_flip = st.slider(
                "Vertical Flip",
                min_value=0.0,
                max_value=1.0,
                value=aug_config['dist']['VerticalFlip'],
                step=0.05
            )
        
        with col3:
            gaussian_blur = st.slider(
                "Gaussian Blur",
                min_value=0.0,
                max_value=1.0,
                value=aug_config['dist']['GaussianBlur'],
                step=0.05
            )
        
        with col4:
            salt_pepper = st.slider(
                "Salt & Pepper",
                min_value=0.0,
                max_value=1.0,
                value=aug_config['dist']['SaltandPepper'],
                step=0.05
            )
        
        # Augmentation Settings
        col1, col2, col3 = st.columns(3)
        with col1:
            aug_cycles = st.number_input(
                "Augmentation Cycles",
                min_value=1,
                max_value=10,
                value=aug_config['cycle']
            )
        
        with col2:
            aug_prob = st.slider(
                "Augmentation Probability",
                min_value=0.0,
                max_value=1.0,
                value=aug_config['file_selection']['prob'],
                step=0.05
            )
        
        with col3:
            train_only_aug = st.checkbox(
                "Apply Augmentation to Training Only",
                value=aug_config['train_only']
            )
        
        # Dataset Split Configuration
        st.markdown('<div class="sub-header">📊 Dataset Split Ratios</div>', unsafe_allow_html=True)
        
        st.markdown("**Object Detection Split**")
        col1, col2, col3 = st.columns(3)
        with col1:
            det_train_ratio = st.slider(
                "Train Ratio",
                min_value=0.1,
                max_value=0.9,
                value=preprocessing_config['preprocessing']['split']['detection_ratio'][0],
                step=0.05,
                key="det_train"
            )
        
        with col2:
            det_val_ratio = st.slider(
                "Validation Ratio",
                min_value=0.05,
                max_value=0.5,
                value=preprocessing_config['preprocessing']['split']['detection_ratio'][1],
                step=0.05,
                key="det_val"
            )
        
        with col3:
            det_test_ratio = 1.0 - det_train_ratio - det_val_ratio
            st.metric("Test Ratio", f"{det_test_ratio:.2f}")
        
        st.markdown("**Classification Split**")
        col1, col2, col3 = st.columns(3)
        with col1:
            cls_train_ratio = st.slider(
                "Train Ratio",
                min_value=0.1,
                max_value=0.9,
                value=preprocessing_config['preprocessing']['split']['classification_ratio'][0],
                step=0.05,
                key="cls_train"
            )
        
        with col2:
            cls_val_ratio = st.slider(
                "Validation Ratio",
                min_value=0.05,
                max_value=0.5,
                value=preprocessing_config['preprocessing']['split']['classification_ratio'][1],
                step=0.05,
                key="cls_val"
            )
        
        with col3:
            cls_test_ratio = 1.0 - cls_train_ratio - cls_val_ratio
            st.metric("Test Ratio", f"{cls_test_ratio:.2f}")
        
        # Update configuration with user inputs
        preprocessing_config['data']['raw_data_dir'] = raw_data_dir
        preprocessing_config['data']['processed_dir'] = processed_dir
        preprocessing_config['data']['images_dir'] = images_dir
        preprocessing_config['data']['labels_dir'] = labels_dir
        preprocessing_config['preprocessing']['resize_dim'] = [resize_width, resize_height]
        preprocessing_config['preprocessing']['augmentation']['dist'] = {
            'HorizontalFlip': horizontal_flip,
            'VerticalFlip': vertical_flip,
            'GaussianBlur': gaussian_blur,
            'SaltandPepper': salt_pepper
        }
        preprocessing_config['preprocessing']['augmentation']['cycle'] = aug_cycles
        preprocessing_config['preprocessing']['augmentation']['file_selection']['prob'] = aug_prob
        preprocessing_config['preprocessing']['augmentation']['train_only'] = train_only_aug
        preprocessing_config['preprocessing']['split']['detection_ratio'] = [det_train_ratio, det_val_ratio, det_test_ratio]
        preprocessing_config['preprocessing']['split']['classification_ratio'] = [cls_train_ratio, cls_val_ratio, cls_test_ratio]
        
        # Run Preprocessing Button
        st.markdown("---")
        if st.button("🚀 Run Data Preprocessing", type="primary", use_container_width=True):
            temp_config_path = save_temp_config(preprocessing_config)
            command = f"python3 src/DpPreprocessing/run_preprocessing.py --config {temp_config_path}"
            run_pipeline_command(command, "Data Preprocessing")
            
            # Cleanup temp file
            try:
                os.remove(temp_config_path)
                shutil.rmtree(os.path.dirname(temp_config_path))
            except:
                pass
    
    # Model Training Module
    elif selected_module == "🚀 Model Training":
        st.markdown('<div class="module-header">🚀 Model Training Configuration</div>', 
                   unsafe_allow_html=True)
        
        # Create a copy of config for modifications
        training_config = config.copy()
        
        # Object Detection Training
        st.markdown('<div class="sub-header">🎯 Object Detection Training</div>', unsafe_allow_html=True)
        
        od_enabled = st.toggle(
            "Enable Object Detection Training",
            value=training_config['training']['object_detection']['enabled']
        )
        
        if od_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                od_model_size = st.selectbox(
                    "Model Size",
                    options=['n', 's', 'm', 'l', 'x'],
                    index=['n', 's', 'm', 'l', 'x'].index(training_config['training']['object_detection']['model']['size']),
                    format_func=lambda x: {
                        'n': 'Nano (Fastest)',
                        's': 'Small (Balanced)',
                        'm': 'Medium (Good)',
                        'l': 'Large (Better)',
                        'x': 'Extra Large (Best)'
                    }[x],
                    key="od_model_size"
                )
                
                od_epochs = st.number_input(
                    "Training Epochs",
                    min_value=1,
                    max_value=1000,
                    value=training_config['training']['object_detection']['epochs'],
                    key="od_epochs"
                )
                
                od_batch_size = st.selectbox(
                    "Batch Size",
                    options=[4, 8, 16, 32, 64],
                    index=[4, 8, 16, 32, 64].index(training_config['training']['object_detection']['batch_size']),
                    key="od_batch_size"
                )
            
            with col2:
                od_learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=training_config['training']['object_detection']['learning_rate'],
                    format="%.4f",
                    key="od_lr"
                )
                
                od_img_size = st.selectbox(
                    "Image Size",
                    options=[320, 416, 512, 640, 832, 1024],
                    index=[320, 416, 512, 640, 832, 1024].index(training_config['training']['object_detection']['img_size']),
                    key="od_img_size"
                )
                
                od_save_dir = st.text_input(
                    "Save Directory",
                    value=training_config['training']['object_detection']['save_dir'],
                    key="od_save_dir"
                )
        
        # Classification Training
        st.markdown('<div class="sub-header">📊 Classification Training</div>', unsafe_allow_html=True)
        
        cls_enabled = st.toggle(
            "Enable Classification Training",
            value=training_config['training']['classification']['enabled']
        )
        
        if cls_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                cls_model_size = st.selectbox(
                    "Model Size",
                    options=['n', 's', 'm', 'l', 'x'],
                    index=['n', 's', 'm', 'l', 'x'].index(training_config['training']['classification']['model']['size']),
                    format_func=lambda x: {
                        'n': 'Nano (Fastest)',
                        's': 'Small (Balanced)',
                        'm': 'Medium (Good)',
                        'l': 'Large (Better)',
                        'x': 'Extra Large (Best)'
                    }[x],
                    key="cls_model_size"
                )
                
                cls_epochs = st.number_input(
                    "Training Epochs",
                    min_value=1,
                    max_value=1000,
                    value=training_config['training']['classification']['epochs'],
                    key="cls_epochs"
                )
                
                cls_batch_size = st.selectbox(
                    "Batch Size",
                    options=[4, 8, 16, 32, 64, 128],
                    index=[4, 8, 16, 32, 64, 128].index(training_config['training']['classification']['batch_size']),
                    key="cls_batch_size"
                )
            
            with col2:
                cls_learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=training_config['training']['classification']['learning_rate'],
                    format="%.4f",
                    key="cls_lr"
                )
                
                cls_img_size = st.selectbox(
                    "Image Size",
                    options=[128, 224, 256, 384, 512],
                    index=[128, 224, 256, 384, 512].index(training_config['training']['classification']['img_size']),
                    key="cls_img_size"
                )
                
                cls_save_dir = st.text_input(
                    "Save Directory",
                    value=training_config['training']['classification']['save_dir'],
                    key="cls_save_dir"
                )
        
        # Update training configuration
        training_config['training']['object_detection']['enabled'] = od_enabled
        if od_enabled:
            training_config['training']['object_detection']['model']['size'] = od_model_size
            training_config['training']['object_detection']['epochs'] = od_epochs
            training_config['training']['object_detection']['batch_size'] = od_batch_size
            training_config['training']['object_detection']['learning_rate'] = od_learning_rate
            training_config['training']['object_detection']['img_size'] = od_img_size
            training_config['training']['object_detection']['save_dir'] = od_save_dir
        
        training_config['training']['classification']['enabled'] = cls_enabled
        if cls_enabled:
            training_config['training']['classification']['model']['size'] = cls_model_size
            training_config['training']['classification']['epochs'] = cls_epochs
            training_config['training']['classification']['batch_size'] = cls_batch_size
            training_config['training']['classification']['learning_rate'] = cls_learning_rate
            training_config['training']['classification']['img_size'] = cls_img_size
            training_config['training']['classification']['save_dir'] = cls_save_dir
        
        # Run Training Button
        st.markdown("---")
        if st.button("🚀 Run Model Training", type="primary", use_container_width=True):
            if not od_enabled and not cls_enabled:
                st.error("Please enable at least one training module (Object Detection or Classification)")
            else:
                temp_config_path = save_temp_config(training_config)
                command = f"python3 src/DpTraining/train.py --config {temp_config_path}"
                run_pipeline_command(command, "Model Training")
                
                # Cleanup temp file
                try:
                    os.remove(temp_config_path)
                    shutil.rmtree(os.path.dirname(temp_config_path))
                except:
                    pass
    
    # Model Evaluation Module
    elif selected_module == "📈 Model Evaluation":
        st.markdown('<div class="module-header">📈 Model Evaluation Configuration</div>', 
                   unsafe_allow_html=True)
        
        # Create a copy of config for modifications
        evaluation_config = config.copy()
        
        # Object Detection Evaluation
        st.markdown('<div class="sub-header">🎯 Object Detection Evaluation</div>', unsafe_allow_html=True)
        
        od_eval_enabled = st.toggle(
            "Enable Object Detection Evaluation",
            value=evaluation_config['evaluation']['object_detection']['enabled']
        )
        
        if od_eval_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                od_model_weights = st.text_input(
                    "Model Weights Path",
                    value=evaluation_config['evaluation']['object_detection']['model_weights'] or "yolov8s.pt",
                    help="Path to trained model weights or default YOLO model",
                    key="od_eval_weights"
                )
                
                od_save_dir = st.text_input(
                    "Results Save Directory",
                    value=evaluation_config['evaluation']['object_detection']['save_dir'],
                    key="od_eval_save_dir"
                )
            
            with col2:
                od_unseen_data_path = st.text_input(
                    "Test Data Path",
                    value=evaluation_config['evaluation']['object_detection']['unseen_data_path'],
                    help="Path to test data (should contain images/ and labels/ folders)",
                    key="od_eval_data_path"
                )
        
        # Classification Evaluation
        st.markdown('<div class="sub-header">📊 Classification Evaluation</div>', unsafe_allow_html=True)
        
        cls_eval_enabled = st.toggle(
            "Enable Classification Evaluation",
            value=evaluation_config['evaluation']['classification']['enabled']
        )
        
        if cls_eval_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                cls_model_weights = st.text_input(
                    "Model Weights Path",
                    value=evaluation_config['evaluation']['classification']['model_weights'] or "yolov8s-cls.pt",
                    help="Path to trained model weights or default YOLO classification model",
                    key="cls_eval_weights"
                )
                
                cls_save_dir = st.text_input(
                    "Results Save Directory",
                    value=evaluation_config['evaluation']['classification']['save_dir'],
                    key="cls_eval_save_dir"
                )
            
            with col2:
                cls_unseen_data_path = st.text_input(
                    "Test Data Path",
                    value=evaluation_config['evaluation']['classification']['unseen_data_path'],
                    help="Path to test data (single folder with mixed class images)",
                    key="cls_eval_data_path"
                )
        
        # Update evaluation configuration
        evaluation_config['evaluation']['object_detection']['enabled'] = od_eval_enabled
        if od_eval_enabled:
            evaluation_config['evaluation']['object_detection']['model_weights'] = od_model_weights
            evaluation_config['evaluation']['object_detection']['unseen_data_path'] = od_unseen_data_path
            evaluation_config['evaluation']['object_detection']['save_dir'] = od_save_dir
        
        evaluation_config['evaluation']['classification']['enabled'] = cls_eval_enabled
        if cls_eval_enabled:
            evaluation_config['evaluation']['classification']['model_weights'] = cls_model_weights
            evaluation_config['evaluation']['classification']['unseen_data_path'] = cls_unseen_data_path
            evaluation_config['evaluation']['classification']['save_dir'] = cls_save_dir
        
        # Evaluation Options
        st.markdown('<div class="sub-header">⚙️ Evaluation Options</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            generate_reports = st.checkbox("Generate Detailed Reports", value=True)
            save_visualizations = st.checkbox("Save Visualizations", value=True)
        
        with col2:
            log_to_mlflow = st.checkbox("Log to MLflow", value=True)
            save_predictions = st.checkbox("Save Predictions", value=True)
        
        # Run Evaluation Button
        st.markdown("---")
        if st.button("🚀 Run Model Evaluation", type="primary", use_container_width=True):
            if not od_eval_enabled and not cls_eval_enabled:
                st.error("Please enable at least one evaluation module (Object Detection or Classification)")
            else:
                temp_config_path = save_temp_config(evaluation_config)
                command = f"python3 src/DpEvaluation/main.py --config {temp_config_path}"
                run_pipeline_command(command, "Model Evaluation")
                
                # Cleanup temp file
                try:
                    os.remove(temp_config_path)
                    shutil.rmtree(os.path.dirname(temp_config_path))
                except:
                    pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 DP Computer Vision Pipeline Demo | Built with Streamlit</p>
        <p>💡 This interface allows you to configure and run the entire ML pipeline without modifying the original config.yaml</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
