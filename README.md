# DP Data Preparation Pipeline for Computer Vision

A comprehensive MLflow-based pipeline for computer vision data preprocessing, training, and evaluation with support for both object detection and classification tasks.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda env create -f conda.yaml
conda activate DP_Pipelines_CV_V1
```

#### Option 2: Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
DP_Data_Prep_Pipline_CV_V1/
├── configs/
│   └── config.yaml                 # Main configuration file
├── data/
│   ├── raw_data/                   # Input raw data
│   └── processed/                  # Processed datasets
├── src/
│   ├── DpPreprocessing/            # Data preprocessing module
│   │   └── run_preprocessing.py    # Preprocessing entry point
│   ├── DpTraining/                 # Model training module
│   │   └── train.py               # Training entry point
│   └── DpEvaluation/              # Model evaluation module
│       └── main.py                # Evaluation entry point
├── pipeline_output/               # Training and evaluation outputs
├── mlruns/                       # MLflow experiment tracking
├── requirements.txt              # Python dependencies
├── conda.yaml                   # Conda environment
└── README.md                     # This file
```

## 🌐 Streamlit Web Interface

For an interactive web-based interface to manage and run the entire pipeline:

```bash
# Launch the Streamlit demo
streamlit run streamlit_demo.py
```

**Features:**
- 🎛️ **Interactive Configuration**: Modify all pipeline parameters through a user-friendly web interface
- 📊 **Real-time Monitoring**: Watch pipeline execution with live output updates
- 🔄 **Module Navigation**: Easy switching between preprocessing, training, and evaluation modules
- 💾 **Non-destructive**: Changes don't modify the original config.yaml file
- 📈 **Visual Feedback**: Progress indicators and success/error notifications

## 🔄 Pipeline Execution Sequence

### 1. Data Preprocessing
Prepares raw data for training by handling format conversion, augmentation, and dataset splitting.

```bash
python3 src/DpPreprocessing/run_preprocessing.py --config configs/config.yaml
```

**What it does:**
- Converts annotations to YOLO format
- Applies data augmentation
- Resizes images
- Creates train/validation/test splits
- Generates classification crops from detection data

### 2. Model Training
Trains both object detection and classification models using YOLO architecture.

```bash
python3 src/DpTraining/train.py --config configs/config.yaml
```

**What it does:**
- Trains YOLOv8 object detection model
- Trains YOLOv8 classification model
- Logs metrics and artifacts to MLflow
- Saves trained model weights

### 3. Model Evaluation
Evaluates trained models on test data with comprehensive metrics and visualizations.

```bash
python3 src/DpEvaluation/main.py --config configs/config.yaml
```

**What it does:**
- **Object Detection**: Computes mAP, precision, recall, IoU metrics
- **Classification**: Computes accuracy, precision, recall, F1-score, confusion matrix
- Generates evaluation reports and visualizations
- Automatically detects data structure and processes accordingly

## ⚙️ Configuration

All pipeline parameters are controlled through `configs/config.yaml`. Key sections:

### Data Configuration
```yaml
data:
  raw_data_dir: raw_data/Data
  processed_dir: pipeline_output/processed_data
  images_dir: Images
  labels_dir: Labels
```

### Training Configuration
```yaml
training:
  object_detection:
    enabled: true
    epochs: 100
    batch_size: 16
  classification:
    enabled: true
    epochs: 50
    batch_size: 32
```

### Evaluation Configuration
```yaml
evaluation:
  object_detection:
    enabled: true
    unseen_data_path: pipeline_output/processed_data/Detection/data/test/
  classification:
    enabled: true
    unseen_data_path: pipeline_output/processed_data/Classification/Split/test/0/
```

## 📊 Data Structure Requirements

### Object Detection
```
unseen_data_path/
├── images/          # Test images
└── labels/          # YOLO format labels
```

### Classification
```
unseen_data_path/    # Single folder with mixed class images
├── image_class0.jpg # Images with class labels in filenames
├── image_class1.jpg
└── ...
```

## 🔍 MLflow Tracking

The pipeline automatically tracks all experiments using MLflow:

```bash
# Start MLflow UI to view experiments
mlflow ui

# Access at http://localhost:5000
```

**Tracked Information:**
- Training metrics (loss, accuracy, mAP)
- Model parameters and hyperparameters
- Dataset artifacts and preprocessed data
- Evaluation results and visualizations
- Model weights and checkpoints

## 📈 Output Structure

### Training Outputs
```
pipeline_output/
├── models/
│   ├── object_detection/
│   │   └── detection_run/
│   │       └── weights/best.pt
│   └── classification/
│       └── classification_run/
│           └── weights/best.pt
```

### Evaluation Outputs
```
pipeline_output/evaluation/
├── object_detection_results/
│   ├── evaluation_metrics.json
│   ├── evaluation_summary.txt
│   └── confusion_matrix.png
└── classification_results/
    ├── evaluation_metrics.json
    ├── evaluation_summary.txt
    ├── confusion_matrix.png
    └── per_class_metrics.png
```

## 🛠️ Key Features

- **🔄 End-to-end Pipeline**: Complete workflow from raw data to trained models
- **📊 MLflow Integration**: Comprehensive experiment tracking and artifact management
- **🎯 Dual Task Support**: Both object detection and classification in one pipeline
- **🔧 Config-driven**: All parameters controlled through YAML configuration
- **📈 Rich Evaluation**: Comprehensive metrics and visualizations
- **🚀 Production Ready**: Robust error handling and logging
- **🔍 Auto Detection**: Automatically detects data structure and processes accordingly

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **CUDA Issues**: Install appropriate PyTorch version for your CUDA version
3. **Memory Issues**: Reduce batch size in config.yaml
4. **Path Issues**: Use absolute paths in config.yaml if relative paths fail

### Getting Help

1. Check MLflow UI for detailed experiment logs
2. Review log files in the terminal output
3. Verify data structure matches requirements
4. Ensure all dependencies are installed correctly

## 📝 Notes

- The pipeline is designed to work with YOLO format annotations
- All modules are independent and can be run separately if needed
- Configuration changes take effect immediately without code modifications
- The evaluation module automatically handles different data structures

## 🔄 Complete Workflow Example

### Option 1: Using Streamlit Web Interface (Recommended)
```bash
# 1. Activate environment
conda activate DP_Pipelines_CV_V1

# 2. Launch interactive web interface
streamlit run streamlit_demo.py

# 3. Configure and run pipeline through the web interface
# 4. View results in MLflow UI
mlflow ui
```

### Option 2: Using Command Line
```bash
# 1. Activate environment
conda activate DP_Pipelines_CV_V1

# 2. Run complete pipeline
python3 src/DpPreprocessing/run_preprocessing.py --config configs/config.yaml
python3 src/DpTraining/train.py --config configs/config.yaml
python3 src/DpEvaluation/main.py --config configs/config.yaml

# 3. View results
mlflow ui
```

That's it! Your complete computer vision pipeline is ready to use. 🎉
