# BTXRD: Bone Tumor X-Ray Detection with Knowledge Distillation

AI system for automatic bone tumor classification and segmentation from X-ray images using deep learning and knowledge distillation.

---

## Project Overview

This project develops lightweight, efficient models for bone tumor diagnosis from X-ray images by applying **Knowledge Distillation (KD)** to compress large teacher models into smaller student models while maintaining high accuracy.

**Key Tasks:**
- **Classification**: Identify tumor type (9 classes)
- **Segmentation**: Locate tumor region in X-ray image

---

## Dataset

- **Total Images**: 1,150 X-ray images with augmentation
- **Classes**: 9 bone tumor types
  - Giant Cell Tumor, Multiple Osteochondromas, Osteochondroma, Osteofibroma
  - Osteosarcoma, Other Benign Tumors, Other Malignant Tumors
  - Simple Bone Cyst, Synovial Osteochondroma
- **Splits**: Train (70%), Val (15%), Test (15%)
- **Annotations**: Classification labels + binary segmentation masks

---

## Phase 1: Baseline Models

### Classification
- **Model**: EfficientNet-B4
- **Parameters**: 19M
- **Test Accuracy**: 85.03%
- **Training**: Standard cross-entropy loss with data augmentation

### Segmentation
- **Model**: U-Net with ResNet-50 encoder
- **Parameters**: 35M
- **Test Dice Score**: 65.12%
- **Training**: Combined Dice + BCE loss

---

## Phase 2: Knowledge Distillation

Applied knowledge distillation to create smaller, faster models that maintain performance.

### Classification KD
- **Teacher**: EfficientNet-B4 (19M params, 85% accuracy)
- **Student**: ConvNeXt-Tiny (28M params)
- **Method**: Response-based KD with temperature scaling (T=4.0)
- **Loss**: 0.5 × KL Divergence + 0.5 × Cross-Entropy
- **Results**: 73.80% test accuracy (11.2% drop from teacher)

**Training Details:**
- Optimizer: AdamW (lr=1e-4, weight decay=0.05)
- Epochs: 30 with early stopping
- Augmentations: RandomHorizontalFlip, RandomRotation, ColorJitter
- Time: ~90 minutes on RTX 4090

### Segmentation KD
- **Teacher**: U-Net ResNet-50 (35M params, 65% Dice)
- **Student**: SegFormer-B2 (28M params)
- **Method**: Response-based KD with feature map distillation
- **Loss**: 0.3 × Distillation + 0.7 × (Dice + BCE)
- **Results**: 50.94% test Dice score (14.2% drop from teacher)

**Training Details:**
- Optimizer: AdamW (lr=5e-5, weight decay=0.01)
- Epochs: 50 with early stopping
- Augmentations: RandomHorizontalFlip, RandomRotation, ElasticTransform
- Time: ~3 hours on RTX 4090

---

## Model Compression Results

| Task | Teacher | Student | Size Reduction | Accuracy Drop |
|------|---------|---------|----------------|---------------|
| **Classification** | EfficientNet-B4 (19M) | ConvNeXt-Tiny (28M) | -47% params* | -11.2% |
| **Segmentation** | U-Net ResNet-50 (35M) | SegFormer-B2 (28M) | -20% params | -14.2% |

*ConvNeXt has more parameters but is optimized for efficiency with depthwise convolutions

---

## Combined Inference System

Unified inference script that runs both classification and segmentation on same images with comprehensive visualizations.

**Features:**
- Processes 187 test images (2.16 it/s on RTX 4090)
- 8-panel visualization per image:
  - Original X-ray, Ground truth mask, Predicted mask, Comparison overlay
  - Segmentation Grad-CAM, Classification Grad-CAM, Top-5 predictions, Summary card
- Metrics CSV with per-sample results
- Poster-quality output (200 DPI)

**Location**: `combined_inference/`

**Performance:**
- Classification: 73.80% accuracy (138/187 correct)
- Segmentation: 51.07% average Dice, 42.73% average IoU

---

## Project Structure

```
BTXRD/
├── augmented_classification_data/    # Dataset with train/val/test splits
├── segmentation_masks/                # Binary masks for all images
├── btxrd-classification-kd/          # Classification KD training code
│   ├── configs/                       # Training configurations
│   ├── models/                        # Teacher & student architectures
│   └── outputs/kd_student/            # Trained student weights
├── btxrd-segmentation-kd/            # Segmentation KD training code
│   ├── configs/                       # Training configurations
│   ├── models/                        # Teacher & student architectures
│   └── outputs/kd_student/            # Trained student weights
├── combined_inference/               # Unified inference system
│   ├── infer.py                       # Main inference script
│   ├── models/                        # Copied student checkpoints
│   └── results/                       # Visualizations + metrics CSV
├── classification/                    # Baseline classification code
├── segmentation/                      # Baseline segmentation code
└── common/                           # Shared utilities (Grad-CAM, metrics)
```

---

## Key Technologies

- **Framework**: PyTorch 2.x
- **Models**: timm (ConvNeXt), transformers (SegFormer)
- **Visualization**: Grad-CAM, Matplotlib
- **Metrics**: Accuracy, Dice Score, IoU, Confusion Matrix
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)

---

## Documentation

- `KNOWLEDGE_DISTILLATION_README.md` - Detailed KD implementation guide
- `PHASE1_BASELINE_REPORT.md` - Baseline model training results
- `FULL_INFERENCE_RESULTS.md` - Complete inference analysis
- `PROJECT_ABSTRACT.md` - Project summary and objectives

---

## Future Work

1. **Performance Improvements**:
   - Experiment with advanced KD techniques (FitNets, Attention Transfer)
   - Train stronger teacher models for better knowledge transfer
   - Hyperparameter tuning (temperature, loss weights)

2. **Model Optimization**:
   - Convert to ONNX/TorchScript for faster inference
   - Quantization (INT8) for deployment on edge devices

3. **Clinical Deployment**:
   - Web-based interface for radiologists
   - Integration with PACS systems
   - Multi-view X-ray analysis

---

## Authors

Nauman - Final Year Project, VistAI Lab

**Last Updated**: February 2026
