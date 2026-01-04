# Evaluation Module Documentation

## Overview

The evaluation module provides a comprehensive, config-driven evaluation system for machine learning models with automatic data type detection. It supports both object detection and classification tasks with extensive metrics computation and visualization generation.

## Features

### 🔍 Automatic Data Type Detection
- **Object Detection**: Automatically detects `images/` and `labels/` folder structure
- **Classification**: Automatically detects numbered class folders (0, 1, 2, etc.)

### 📊 Comprehensive Metrics

#### Object Detection Metrics
- **mAP (mean Average Precision)**: mAP@0.5, mAP@0.75, mAP@0.5:0.95
- **Precision & Recall**: Mean precision and recall across all classes
- **Per-class metrics**: Individual mAP scores for each class
- **Performance metrics**: Inference speed and timing information

#### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Both macro and weighted averages
- **Per-class metrics**: Individual precision, recall, and F1-score for each class
- **Confusion Matrix**: Both raw counts and normalized versions

### 📈 Visualizations & Reports
- Confusion matrices (raw and normalized)
- Per-class performance charts
- Comprehensive evaluation summaries
- JSON reports for programmatic access
- MLflow integration for experiment tracking

## Usage

### Command Line Interface

```bash
python3 src/DpEvaluation/main.py --config configs/config.yaml
```

### Configuration

The evaluation module is entirely config-driven. Update your `config.yaml` file:

```yaml
evaluation:
  object_detection:
    enabled: true
    model_weights: pipeline_output/models/object_detection/detection_run/weights/best.pt
    unseen_data_path: path/to/your/test/data  # Must contain images/ and labels/ folders
    save_dir: pipeline_output/evaluation/object_detection
  classification:
    enabled: true
    model_weights: pipeline_output/models/classification/classification_run/weights/best.pt
    unseen_data_path: path/to/your/test/data  # Must contain class folders: 0/, 1/, 2/, etc.
    save_dir: pipeline_output/evaluation/classification
```

## Data Structure Requirements

### Object Detection Data Structure
```
unseen_data_path/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Classification Data Structure
```
unseen_data_path/
├── 0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 1/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
├── 2/
│   └── ...
└── ...
```

## Output Structure

### Object Detection Results
```
{save_dir}/object_detection_results/
├── evaluation_metrics.json          # All metrics in JSON format
├── evaluation_summary.txt           # Human-readable summary
├── confusion_matrix.png            # Confusion matrix visualization
├── temp_eval_data.yaml             # Temporary YOLO data file (auto-cleaned)
└── [YOLO validation outputs]       # Additional YOLO-generated files
```

### Classification Results
```
{save_dir}/classification_results/
├── evaluation_metrics.json          # All metrics in JSON format
├── evaluation_summary.txt           # Human-readable summary
├── classification_report.json       # Detailed sklearn classification report
├── confusion_matrix.png            # Raw confusion matrix
├── confusion_matrix_normalized.png  # Normalized confusion matrix
└── per_class_metrics.png           # Per-class performance chart
```

## Key Features

### 1. Automatic Data Type Detection
The module automatically detects whether your data is for object detection or classification:

```python
from DpEvaluation.utils import detect_data_type

data_type = detect_data_type("/path/to/your/data")
# Returns: 'object_detection', 'classification', or None
```

### 2. Comprehensive Error Handling
- Validates all configuration paths before execution
- Provides detailed error messages and logging
- Graceful handling of missing files or incorrect data structures

### 3. MLflow Integration
- Automatic experiment tracking
- Metrics logging to MLflow
- Artifact storage for all generated reports and visualizations

### 4. Flexible Model Loading
- Automatically finds trained models if `model_weights` is not specified
- Falls back to latest training outputs
- Supports custom model weight paths

## Configuration Parameters

### Required Parameters
- `enabled`: Boolean to enable/disable evaluation for each task type
- `unseen_data_path`: Path to test data (must follow required structure)
- `save_dir`: Directory to save evaluation results

### Optional Parameters
- `model_weights`: Path to specific model weights (auto-detected if not provided)

## Error Handling

The module includes comprehensive error handling for common issues:

1. **Missing Configuration**: Clear error messages for missing required fields
2. **Invalid Data Paths**: Validation of all specified paths
3. **Incorrect Data Structure**: Automatic detection with helpful error messages
4. **Model Loading Issues**: Fallback mechanisms and clear error reporting
5. **Processing Errors**: Graceful handling with detailed logging

## Logging

The module uses structured logging with different levels:
- **INFO**: General progress and status updates
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical issues that prevent execution
- **DEBUG**: Detailed information for troubleshooting

## Integration with Existing Pipeline

The evaluation module is designed to integrate seamlessly with your existing preprocessing and training modules:

1. **No modifications required** to preprocessing or training code
2. **Config-driven approach** ensures consistency across pipeline stages
3. **Automatic model discovery** from training outputs
4. **MLflow integration** maintains experiment continuity

## Example Workflow

1. **Train your models** using the existing training pipeline
2. **Prepare test data** in the required structure
3. **Update config.yaml** with test data paths
4. **Run evaluation**:
   ```bash
   python3 src/DpEvaluation/main.py --config configs/config.yaml
   ```
5. **Review results** in the specified output directories
6. **Check MLflow** for logged metrics and artifacts

## Troubleshooting

### Common Issues

1. **"Could not detect data type"**
   - Ensure your data follows the required folder structure
   - Check that folders contain the expected files (images, labels, etc.)

2. **"Model weights not found"**
   - Verify the model_weights path in config.yaml
   - Ensure training has completed successfully
   - Check that the training output directory contains the expected structure

3. **"Unseen data path does not exist"**
   - Verify the unseen_data_path in config.yaml
   - Ensure the path is accessible from the project root

### Getting Help

For additional support:
1. Check the logs for detailed error messages
2. Verify your data structure matches the requirements
3. Ensure all dependencies are installed (see requirements.txt)
4. Review the configuration file for any missing or incorrect paths

## Dependencies

The evaluation module requires the following packages:
- ultralytics (YOLO models)
- scikit-learn (metrics computation)
- matplotlib & seaborn (visualizations)
- numpy & pandas (data processing)
- mlflow (experiment tracking)
- PyYAML (configuration parsing)

All dependencies should already be available in your existing environment.
