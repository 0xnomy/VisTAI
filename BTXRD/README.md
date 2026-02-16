# 🩻 Bone Tumor X-Ray Diagnosis System (BTXRD)

**Multi-Task Deep Learning System for Bone Tumor Segmentation and Classification**

> **Project Development in Progress** 

## 🎯 Overview

This project implements a comprehensive deep learning pipeline for automated bone tumor diagnosis from X-ray images. The system consists of three main components:

1. **Tumor Segmentation**: Pixel-level identification of tumor regions
2. **Tumor Classification**: Multi-class classification of bone tumors into 9 categories
3. **Knowledge Distillation**: Model compression for efficient deployment

### Key Features

- ✅ **Advanced Deep Learning Models** for segmentation and classification
- ✅ **Knowledge Distillation Pipeline** for model compression and deployment
- ✅ **Comprehensive Training Framework** with multiple architectures
- ✅ **Interactive Frontend** with multimodal capabilities
- ✅ **Detailed Performance Analysis** and visualization tools

---

## 📊 Dataset

- **Segmentation Dataset**: 1,867 X-ray images with tumor masks
- **Classification Dataset**: 8,958 images across 9 tumor categories
- **Data Split**: 80/10/10 train/validation/test

---

## 🏗️ Project Structure

```
BTXRD/
├── segmentation/                    # Tumor Segmentation Module
├── classification/                  # Tumor Classification Module
├── btxrd-segmentation-kd/          # Knowledge Distillation for Segmentation
├── btxrd-classification-kd/        # Knowledge Distillation for Classification
├── common/                          # Shared utilities and tools
├── frontend/                        # Interactive UI and chat interface
├── multimodal/                      # Multimodal pipeline components
└── augmented_classification_data/   # Augmented training data
```

---

## 🚀 Current Progress

### ✅ Phase 1: Teacher Models (Baseline)
- [x] Segmentation teacher model trained and evaluated
- [x] Classification teacher model trained and evaluated
- [x] Performance analysis and failure analysis completed
- [x] Baseline metrics established

### 🔄 Phase 2: Knowledge Distillation (In Progress)
- [x] Segmentation KD framework implemented
- [x] Classification KD framework implemented
- [x] Student model architectures defined
- [x] Training pipelines configured
- [ ] Full training runs and optimization
- [ ] Comparative performance analysis

### 📋 Phase 3: Integration & Deployment (Upcoming)
- [ ] Multimodal pipeline integration
- [ ] Frontend interface finalization
- [ ] Model optimization and quantization
- [ ] Deployment configuration

---

## 📈 Key Results

**Segmentation Teacher Model:**
- Successfully segments medium to large tumors
- Established baseline performance metrics

**Classification Teacher Model:**
- Multi-class classification across 9 tumor types
- Handles class imbalance effectively

**Knowledge Distillation:**
- Framework implemented for both tasks
- Target: Maintain performance with smaller models

*Detailed results available in dedicated documentation files*

---

## 📚 Documentation

- [Knowledge Distillation Overview](KNOWLEDGE_DISTILLATION_README.md)
- [Phase 1 Baseline Report](PHASE1_BASELINE_REPORT.md)
- [Full Inference Results](FULL_INFERENCE_RESULTS.md)
- Module-specific READMEs in respective directories

---

## 🔧 Technology Stack

- **Deep Learning**: PyTorch, torchvision, timm
- **Computer Vision**: OpenCV, Pillow, Albumentations
- **Visualization**: Matplotlib, GradCAM
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy

---

## 📝 License

Private project - All rights reserved

---

## 👥 Contributors

Private project in development

---

*Last Updated: January 2026*
