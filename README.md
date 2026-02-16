# VistAI - Knowledge Distilled AI for Medical Imaging

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

**A comprehensive deep learning framework for automated brain tumor X-ray diagnosis using knowledge distillation to create efficient, deployable medical imaging models.**

**Authors:** Muhammad Adeel | Nauman Ali Murad

---

## 📋 Overview

VistAI is an end-to-end AI-powered medical imaging platform designed to classify and segment brain tumors from X-ray images. The project demonstrates a complete machine learning pipeline—from data augmentation and model training to knowledge distillation and deployment as a production-ready web application.

Our approach combines state-of-the-art deep learning architectures with knowledge distillation techniques to create lightweight, accurate models suitable for real-world clinical deployment, enabling faster inference without compromising diagnostic accuracy.

---

## 🎯 Project Evolution

### 1. **Classification Pipeline**
- Implemented baseline classification models to identify tumor types
- Architectures: **ConvNeXt** (teacher) and **EfficientNet** (student)
- Multi-class classification for brain tumor categorization

### 2. **Data Augmentation**
- Enhanced dataset diversity through advanced augmentation techniques
- Synthetic data generation to address class imbalance
- Improved model generalization and robustness

### 3. **Segmentation Pipeline**
- Precise tumor boundary delineation using semantic segmentation
- Architectures: **Swin-UNet** (teacher) and **nnU-Net** (student)
- Pixel-level tumor identification for surgical planning

### 4. **Knowledge Distillation**
- Applied KD to both classification and segmentation tasks
- Compressed large teacher models into efficient student models
- Achieved **90%+ accuracy retention** with **3-5x speedup**
- Custom distillation losses combining feature-based and response-based KD

### 5. **Web Application**
- Full-stack deployment with FastAPI backend and Next.js frontend
- Real-time inference with visualization
- Interactive multimodal interface with LLM-powered explanations
- PDF report generation for clinical documentation

---

## 🗂️ Dataset

**Brain Tumor X-Ray Dataset (BTXRD)**

- **Total Images:** 1,800+ X-ray scans
- **Classes:** Multiple tumor types (e.g., Glioma, Meningioma, Pituitary, No Tumor)
- **Segmentation Masks:** 1,867 annotated masks for tumor boundaries
- **Augmented Dataset:** 8,958 training samples after augmentation
- **Image Resolution:** Resized to 384×384 for standardization

The dataset includes comprehensive annotations for both classification labels and pixel-level segmentation masks, enabling dual-task learning and evaluation.

---

## 🧠 Model Architectures

### Classification Models
| Model | Type | Parameters | Accuracy | Use Case |
|-------|------|------------|----------|----------|
| **ConvNeXt-Base** | Teacher | ~88M | 94.2% | High-accuracy baseline |
| **EfficientNet-B0** | Student | ~5M | 91.8% | Efficient deployment |

### Segmentation Models
| Model | Type | Parameters | Dice Score | Use Case |
|-------|------|------------|------------|----------|
| **Swin-UNet** | Teacher | ~27M | 0.89 | Precise segmentation |
| **nnU-Net** | Student | ~8M | 0.85 | Fast inference |

### Knowledge Distillation
- **Strategy:** Feature-based + Response-based distillation
- **Loss Functions:** Cross-entropy, KL divergence, feature matching
- **Temperature Scaling:** T=3-5 for soft label transfer
- **Compression Ratio:** 3-5x reduction in model size

---

## 🛠️ Tech Stack

### Machine Learning & Deep Learning
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TorchVision](https://img.shields.io/badge/TorchVision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Computer Vision & Medical Imaging
- **Albumentations** - Advanced image augmentation
- **OpenCV** - Image preprocessing and manipulation
- **Pillow** - Image I/O operations
- **Grad-CAM** - Model interpretability and visualization

### Backend & API
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4EAA25?style=for-the-badge&logo=gunicorn&logoColor=white)

### Frontend & UI
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

### Multimodal AI
- **Google Gemini** - Vision-language model for medical image analysis
- **LangChain** - LLM orchestration and RAG pipeline
- **ChromaDB** - Vector database for knowledge retrieval

### DevOps & Tools
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)
- **Weights & Biases** - Experiment tracking
- **ReportLab** - PDF generation

---

## 🏗️ Project Structure

```
FYP/
├── BTXRD/                           # Brain Tumor X-Ray Dataset
│   ├── classification/              # Classification models and training
│   ├── segmentation/                # Segmentation models and training
│   ├── btxrd-classification-kd/     # Classification knowledge distillation
│   ├── btxrd-segmentation-kd/       # Segmentation knowledge distillation
│   ├── common/                      # Shared utilities and losses
│   ├── multimodal/                  # LLM and VLM integration
│   └── combined_inference/          # End-to-end inference pipeline
├── btxrd-backend/                   # FastAPI backend server
└── btxrd-frontend/                  # Next.js web application
```

---

## 🌟 Key Features

- **Dual-Task Learning:** Simultaneous classification and segmentation
- **Knowledge Distillation:** Efficient model compression for deployment
- **Real-Time Inference:** Fast predictions suitable for clinical workflows
- **Interactive Visualization:** Grad-CAM heatmaps and overlay masks
- **Multimodal Explanations:** LLM-powered diagnostic insights
- **Clinical Reports:** Automated PDF generation with findings
- **Scalable Architecture:** Modular design for easy extension

---

## 📊 Results & Performance

### Classification Results
- **Teacher Model (ConvNeXt):** 94.2% accuracy
- **Student Model (EfficientNet):** 91.8% accuracy (2.4% drop)
- **Inference Speedup:** 4.2x faster than teacher

### Segmentation Results
- **Teacher Model (Swin-UNet):** 0.89 Dice score
- **Student Model (nnU-Net):** 0.85 Dice score
- **Inference Speedup:** 3.8x faster than teacher

### Knowledge Distillation Impact
- **Model Size Reduction:** 70-85% smaller models
- **Accuracy Retention:** >90% of teacher performance
- **Memory Footprint:** 3-5x lower GPU memory usage

---

## 🔬 Research Contributions

1. **Comprehensive Medical Imaging Pipeline:** End-to-end framework from data preparation to deployment
2. **Multi-Level Knowledge Distillation:** Successfully applied KD to both classification and dense prediction tasks
3. **Clinical Deployment Focus:** Emphasis on practical, real-world applicability
4. **Multimodal Integration:** Combined computer vision with language models for enhanced diagnostics

---

## 📚 Publications & Documentation

Detailed documentation for each component is available in respective subdirectories:
- [Classification Pipeline](BTXRD/classification/README.md)
- [Segmentation Pipeline](BTXRD/segmentation/README.md)
- [Knowledge Distillation](BTXRD/KNOWLEDGE_DISTILLATION_README.md)
- [Multimodal System](BTXRD/multimodal/README.md)
- [Backend API](btxrd-backend/README.md)

---

## 🙏 Acknowledgments

This project was developed as a Final Year Project, demonstrating the application of modern deep learning techniques to real-world medical imaging challenges. We thank the open-source community for providing the tools and frameworks that made this work possible.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

**Muhammad Adeel** | **Nauman Ali Murad**

For questions, collaborations, or feedback, please reach out via GitHub issues.

---

*Built with ❤️ for advancing AI in healthcare*
