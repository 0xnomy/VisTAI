# VistAI – Knowledge-Distilled AI for Medical Imaging

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

**An AI-driven medical imaging system for brain tumor classification and segmentation from X-ray images, leveraging knowledge distillation to enable efficient edge deployment.**

**Authors:** Muhammad Adeel | Nauman Ali Murad  

---

## Overview

VistAI is an end-to-end medical imaging system designed for automated brain tumor classification and segmentation from X-ray scans. The system integrates established deep learning architectures with knowledge distillation (KD) techniques to produce computationally efficient models suitable for real-world and edge-device deployment.

This project does not introduce a novel deep learning framework. Instead, it focuses on the systematic application of knowledge distillation to compress high-capacity teacher models into lightweight student models while preserving diagnostic performance. The resulting models demonstrate reduced inference latency, lower memory footprint, and improved deployability without significant accuracy degradation.

---

## System Development Phases

### 1. Classification Pipeline
- Multi-class brain tumor classification  
- Teacher–student training paradigm  
- High-capacity backbone used as teacher  
- Lightweight architecture optimized as student  
- Focus on balancing predictive accuracy and computational efficiency  

### 2. Data Augmentation Strategy
- Advanced augmentation techniques to enhance robustness  
- Mitigation of class imbalance through synthetic transformations  
- Improved generalization across validation datasets  

### 3. Segmentation Pipeline
- Pixel-level tumor boundary delineation  
- Teacher–student distillation for dense prediction  
- Optimization for real-time inference  

### 4. Knowledge Distillation Strategy
- Applied to both classification and segmentation tasks  
- Combined response-based and feature-based distillation  
- KL divergence with temperature scaling for soft label transfer  
- Feature alignment losses for representation matching  
- Significant reduction in model size and computational cost  

### 5. Deployment Architecture
- Backend inference service using FastAPI  
- Frontend interface developed with Next.js  
- Real-time prediction visualization  
- Automated PDF report generation for clinical-style summaries  

---

## Dataset

**Brain Tumor X-Ray Dataset (BTXRD)**

- Total Images: 1,800+ X-ray scans  
- Classes: Glioma, Meningioma, Pituitary, No Tumor  
- Segmentation Masks: 1,867 annotated masks  
- Augmented Samples: 8,958 training instances  
- Standardized Resolution: 384×384 pixels  

The dataset supports both classification and segmentation tasks, enabling multi-task evaluation.

---

## Model Architectures

### Classification Models

| Model | Role | Parameters | Accuracy |
|-------|------|------------|----------|
| ConvNeXt-Base | Teacher | ~88M | 94.2% |
| EfficientNet-B0 | Student | ~5M | 91.8% |

### Segmentation Models

| Model | Role | Parameters | Dice Score |
|-------|------|------------|------------|
| Swin-UNet | Teacher | ~27M | 0.89 |
| nnU-Net | Student | ~8M | 0.85 |

---

## Knowledge Distillation Configuration

- Distillation Type: Feature-based + Response-based  
- Temperature Scaling: T = 3–5  
- Loss Functions:
  - Cross-Entropy Loss  
  - KL Divergence  
  - Feature Matching Loss  
- Model Compression: 70–85% parameter reduction  
- Inference Acceleration: 3–5× speed improvement  

---

## Technical Stack

### Machine Learning
- PyTorch  
- TorchVision  
- NumPy  
- Pandas  
- scikit-learn  

### Computer Vision
- Albumentations  
- OpenCV  
- Pillow  
- Grad-CAM  

### Backend
- FastAPI  
- Uvicorn  

### Frontend
- Next.js  
- TypeScript  
- TailwindCSS  

### Multimodal Integration
- Vision-Language Model Integration  
- LLM-based explanation pipeline  
- Vector database for contextual retrieval  

### Experimentation & Tools
- Weights & Biases  
- Git  
- YAML configuration  
- ReportLab (PDF generation)  

---

## Key Features

- Dual-task learning (classification + segmentation)  
- Knowledge-distilled lightweight models  
- Edge-device deployment readiness  
- Real-time inference  
- Visual interpretability (Grad-CAM & overlays)  
- AI-assisted diagnostic summaries  
- Automated PDF clinical reporting  

---

## Performance Summary

### Classification
- Teacher Accuracy: 94.2%  
- Student Accuracy: 91.8%  
- Accuracy Retention: >90% of teacher performance  
- Inference Speedup: 4×  

### Segmentation
- Teacher Dice Score: 0.89  
- Student Dice Score: 0.85  
- Inference Speedup: ~3.8×  

### Deployment Impact
- 70–85% reduction in model size  
- 3–5× reduction in GPU memory usage  
- Suitable for constrained hardware environments  

---

## Research Contributions

1. Systematic application of knowledge distillation to both classification and dense prediction tasks in medical imaging  
2. Demonstration of high accuracy retention under aggressive model compression  
3. Deployment-oriented medical AI pipeline  
4. Integration of computer vision with language-based explanation modules  

---

## License

This project is licensed under the MIT License.

---

## Contact

**Muhammad Adeel**  
**Nauman Ali Murad**

For questions or collaboration inquiries, please open an issue on GitHub.
