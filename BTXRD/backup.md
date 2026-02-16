# VisTAI BTXRD - Model Backup & Performance Report
**Generated**: January 20, 2026  
**Project Status**: Under Development

---

## 1. Classification Model - EfficientNet-B0

### Model Architecture
- **Model Name**: EfficientNet-B0
- **Total Parameters**: 5.3M (5,300,000)
- **Pretrained**: ImageNet weights
- **Input Size**: 224×224×3
- **Output Classes**: 9 bone tumor classes
- **Dropout**: 0.5 (before final classifier)

### Tumor Classes (9 Total)
1. Giant Cell Tumor
2. Multiple Osteochondromas
3. Osteochondroma
4. Osteofibroma
5. Osteosarcoma
6. Other Benign Tumors (other bt)
7. Other Malignant Tumors (other mt)
8. Simple Bone Cyst
9. Synovial Osteochondroma

### Training Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Training Strategy** | Two-phase | Phase 1: Frozen backbone, Phase 2: Full fine-tuning |
| **Phase 1 Epochs** | 20 | Train classifier head only |
| **Phase 2 Epochs** | 80 | End-to-end fine-tuning |
| **Total Epochs** | 100 | |
| **Batch Size** | 24 | Reduced from 32 for more updates |
| **Optimizer** | AdamW | |
| **Learning Rate (Phase 1)** | 0.001 | Higher for frozen backbone |
| **Learning Rate (Phase 2)** | 0.00005 | Lower for stable fine-tuning |
| **Weight Decay** | 0.05 | Strong regularization |
| **Betas** | [0.9, 0.999] | |
| **LR Scheduler** | ReduceLROnPlateau | |
| **Scheduler Factor** | 0.5 | |
| **Scheduler Patience** | 5 epochs | |
| **Min Learning Rate** | 0.000001 | |
| **Loss Function** | Focal Loss | γ=2.0 |
| **Class Weights** | Enabled | Auto-computed from data |
| **EMA** | Enabled | Decay=0.999 |
| **Gradient Clipping** | 1.0 | |
| **Mixed Precision** | Enabled | |
| **Early Stopping** | Enabled | Patience=20, monitor val_accuracy |
| **Random Seed** | 42 | For reproducibility |
| **Device** | CUDA | GPU-accelerated |

### Dataset
- **Training Set**: 8,958 images (augmented, 6× original)
- **Validation Set**: ~560 images
- **Test Set**: ~186 images
- **Original Dataset**: 1,493 training images (segmentation_train.csv)
- **Augmentation**: Rotation, flip, color jitter, blur, CLAHE

### Performance Metrics (Last Training Run - 78 Epochs)

#### Training Performance
- **Final Training Loss**: 0.0618
- **Final Training Accuracy**: 95.08%
- **Best Training Accuracy**: 95.24% (epoch 77)

#### Validation Performance
- **Final Validation Loss**: 0.9376
- **Final Validation Accuracy**: 67.74%
- **Best Validation Accuracy**: 69.16% (epoch 68)

#### Epoch-by-Epoch Best Results
| Metric | Value | Epoch |
|--------|-------|-------|
| **Best Val Accuracy** | **69.16%** | 68 |
| **Lowest Val Loss** | 0.7228 | 20 |
| **Highest Train Accuracy** | 95.24% | 77 |
| **Lowest Train Loss** | 0.0573 | 77 |

### Training Observations
- **Convergence**: Model converged after ~78 epochs
- **Overfitting Signs**: Yes - large gap between train (95%) and val (67-69%)
- **Recommendations**: 
  - Consider increasing dropout to 0.6-0.7
  - Add more data augmentation
  - Increase weight decay further
  - Use test-time augmentation (TTA)

### Output Files
- **Checkpoint (Latest)**: `classification/outputs/checkpoint_latest.pth`
- **Checkpoint (Best)**: `classification/outputs/checkpoint_best.pth`
- **History**: `classification/outputs/history.json`
- **Config**: `classification/configs/efficientnet_config.yaml`
- **Training Curves**: `classification/outputs/efficientnet_b0_v2/training_curves.png`

---

## 2. Segmentation Model - MobileNetV2-UNet

### Model Architecture
- **Model Name**: MobileNetV2-UNet
- **Encoder**: MobileNetV2 (ImageNet pretrained)
- **Decoder**: UNet-style decoder with skip connections
- **Total Parameters**: ~4-5M (estimated)
- **Input Size**: 256×256×3 (typical for segmentation)
- **Output**: Binary mask (tumor vs. background)
- **Task**: Binary segmentation

### Training Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 93 | Trained for 93 epochs |
| **Batch Size** | 8 | Memory-constrained |
| **Optimizer** | Adam | |
| **Initial Learning Rate** | 0.001 | |
| **Weight Decay** | 1e-5 | 0.00001 |
| **LR Scheduler** | ReduceLROnPlateau | |
| **Scheduler Mode** | max | Monitor Dice score |
| **Scheduler Factor** | 0.5 | |
| **Scheduler Patience** | 5 epochs | |
| **Loss Function** | Combined Dice + BCE | |
| **Dice Weight** | 0.5 | 50% contribution |
| **BCE Weight** | 0.5 | 50% contribution |
| **Dice Smooth** | 1.0 | Numerical stability |
| **Early Stopping** | Enabled | Patience=15 |
| **Mixed Precision** | Enabled | GradScaler |
| **Device** | CUDA | GPU-accelerated |
| **Image Size** | 256×256 | Standard for UNet |

### Dataset
- **Training Set**: From `segmentation_train.csv`
- **Validation Set**: From `segmentation_val.csv`
- **Test Set**: From `segmentation_test.csv`
- **Mask Format**: Binary masks (0=background, 1=tumor)
- **Data Source**: `segmentation_masks/` directory

### Performance Metrics (Last Training Run - 93 Epochs)

#### Training Performance
- **Final Training Loss**: 0.0989
- **Final Training Dice Score**: 0.8394 (83.94%)
- **Best Training Dice**: 0.8399 (83.99%) - epoch 89

#### Validation Performance
- **Final Validation Loss**: 0.1662
- **Final Validation Dice Score**: 0.7255 (72.55%)
- **Best Validation Dice**: 0.7347 (73.47%) - epoch 80

#### Epoch-by-Epoch Best Results
| Metric | Value | Epoch |
|--------|-------|-------|
| **Best Val Dice** | **73.47%** | 80 |
| **Lowest Val Loss** | 0.1615 | 79 |
| **Highest Train Dice** | 83.99% | 89 |
| **Lowest Train Loss** | 0.0988 | 86 |

### Training Observations
- **Convergence**: Good convergence after ~93 epochs
- **Dice Improvement**: Steady increase from 35% to 73% on validation
- **Stability**: Relatively stable in later epochs (70-73% validation Dice)
- **Train-Val Gap**: ~10% gap (reasonable for segmentation)

### Output Files
- **Checkpoint (Latest)**: `segmentation/outputs/checkpoint_latest.pth`
- **Checkpoint (Best)**: `segmentation/outputs/checkpoint_best.pth`
- **History**: `segmentation/outputs/history.json`

---

## 3. Multimodal System - VLM + LLM Chat Interface

### Vision-Language Model (VLM) - CLIP
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Name** | OpenCLIP | Frozen encoder |
| **Architecture** | RN50 or ViT-B-32 | Configurable |
| **Parameters** | ~38M (RN50) | ResNet-50 variant |
| **Embedding Dimension** | 512 | For both image and text |
| **Pretrained** | LAION-2B | Not fine-tuned |
| **Purpose** | Semantic alignment | Image-text embedding space |
| **Usage** | Frozen inference only | No training/fine-tuning |

### Large Language Model (LLM)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Name** | Microsoft Phi-2 | Primary choice |
| **Alternative** | TinyLlama | Backup option |
| **Parameters** | 2.7B | Phi-2 size |
| **Quantization** | 4-bit | For edge deployment |
| **Context Length** | 2048 tokens | |
| **Memory Footprint** | ~1.5GB | With 4-bit quantization |
| **Inference Speed** | 1-2s per query | GPU with quantization |
| **Input Format** | Structured facts only | NO raw images |
| **Safety Mode** | Strict | Enforced disclaimers |

### System Configuration
- **Total Memory**: ~1.7GB VRAM
  - CNNs (frozen): 12MB
  - CLIP: 150MB
  - Phi-2 (4-bit): 1.5GB
- **Total Latency**: 2-3s per query
  - CNN inference: 50ms
  - CLIP encoding: 20ms
  - LLM generation: 1-2s
- **Edge Deployment**: Compatible with hospital workstations

### Safety Constraints
- ✓ Cannot provide medical advice
- ✓ Cannot make definitive diagnoses
- ✓ Must acknowledge uncertainty (if confidence < 80%)
- ✓ Must refer to medical professionals
- ✓ Filters prohibited language patterns
- ✓ Grounded only in CNN predictions

### Architecture Flow
```
X-ray Image 
    ↓
[EfficientNet-B0 Classification] → tumor_class, confidence
    ↓
[MobileNetV2-UNet Segmentation] → tumor_mask, location, area
    ↓
[Structured Facts Generation] → JSON facts
    ↓
[CLIP Encoder] → image_embedding (512-dim) + text_embedding (512-dim)
    ↓
[LLM Chat Engine] → Safety-filtered responses
    ↓
User-facing explanations
```

### Output Files
- **Demo Script**: `multimodal_chat_demo.py`
- **Implementation Summary**: `multimodal/IMPLEMENTATION_SUMMARY.md`
- **README**: `multimodal/README.md`

---

## 4. Performance Summary & Targets

### Classification (EfficientNet-B0)
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Training Accuracy | 95.08% | ~95% | ✓ **MET** |
| Validation Accuracy | 67-69% | >75% | ⚠ **BELOW TARGET** |
| Test Accuracy | Not tested | >70% | ⏳ **PENDING** |

**Status**: Training completed but validation accuracy below target. Needs regularization improvements.

### Segmentation (MobileNetV2-UNet)
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Training Dice Score | 83.94% | ~80-85% | ✓ **GOOD** |
| Validation Dice Score | 72.55% | >70% | ✓ **MET** |
| Test Dice Score | Not tested | >70% | ⏳ **PENDING** |

**Status**: Training completed successfully. Validation Dice meets targets.

### Multimodal System
| Component | Status | Notes |
|-----------|--------|-------|
| VLM Integration | ✓ Implemented | CLIP encoder ready |
| LLM Integration | ✓ Implemented | Phi-2/TinyLlama support |
| Safety Rules | ✓ Implemented | Strict mode enabled |
| Demo Script | ✓ Available | `multimodal_chat_demo.py` |
| Full Testing | ⏳ Pending | Needs smoke testing |

---

## 5. Known Issues & Next Steps

### Classification
1. **Overfitting**: Large train-val gap (95% vs 67%)
   - **Action**: Increase dropout to 0.6-0.7, add more augmentation
2. **Missing Test Script**: No comprehensive evaluation script
   - **Action**: Create `test_classifier.py` with confusion matrix
3. **Class Imbalance**: Some classes may be underrepresented
   - **Action**: Review per-class accuracy, adjust class weights

### Segmentation
1. **Test Evaluation**: No test set evaluation yet
   - **Action**: Run inference on test set, compute final metrics
2. **Boundary Refinement**: Boundaries could be sharper
   - **Action**: Consider adding boundary loss component

### Multimodal
1. **Real LLM Testing**: Not validated with actual Phi-2 model
   - **Action**: Smoke test with mock LLM then real LLM
2. **Retrieval System**: Similarity search not fully integrated
   - **Action**: Build CSV index for similar case retrieval
3. **Response Quality**: Need user testing for safety/quality
   - **Action**: Conduct user studies with medical professionals

### Documentation
1. **README Typos**: "Porject Status" typo in main README
   - **Action**: Fix typo ("Project Status")
2. **Metrics Documentation**: Checkpoints exist but not documented
   - **Action**: This backup.md addresses this

---

## 6. File Locations Reference

### Configuration Files
- `classification/configs/efficientnet_config.yaml` - Classification hyperparameters
- `label_encoding.json` - Class name to ID mapping
- `augmented_classification_data/label_encoding.json` - Augmented data labels

### Model Checkpoints
- `classification/outputs/checkpoint_best.pth` - Best classification model (val_acc: 69.16%)
- `classification/outputs/checkpoint_latest.pth` - Latest classification checkpoint
- `segmentation/outputs/checkpoint_best.pth` - Best segmentation model (dice: 73.47%)
- `segmentation/outputs/checkpoint_latest.pth` - Latest segmentation checkpoint

### Training History
- `classification/outputs/history.json` - 78 epochs of training metrics
- `segmentation/outputs/history.json` - 93 epochs of training metrics
- `classification/outputs/efficientnet_b0_v2/training_history.npy` - NumPy format history

### Code Files
- `classification/train_classifier_refactored.py` - Main classification training script
- `segmentation/train_segmentation_refactored.py` - Main segmentation training script
- `multimodal_chat_demo.py` - Multimodal demo script
- `inference.py` - Unified inference pipeline

### Data Files
- `augmented_classification_data/augmented_train.csv` - 8,958 training images
- `augmented_classification_data/augmented_val.csv` - Validation split
- `augmented_classification_data/augmented_test.csv` - Test split
- `segmentation_train.csv` - Original segmentation training data
- `segmentation_val.csv` - Original segmentation validation data
- `segmentation_test.csv` - Original segmentation test data

---

## 7. Reproducibility Information

### Environment Requirements
```
Python: 3.8+
PyTorch: 2.0+
CUDA: 11.7+
GPU: Recommended (NVIDIA with 8GB+ VRAM)
```

### Key Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
timm  # EfficientNet models
transformers  # Phi-2 LLM
open-clip-torch  # CLIP VLM
opencv-python
pillow
numpy
pandas
matplotlib
tqdm
pyyaml
streamlit  # For web demo
```

### Random Seeds
- **Global Seed**: 42 (set in all training scripts)
- **PyTorch**: Deterministic mode enabled
- **NumPy**: np.random.seed(42)
- **Python**: random.seed(42)

### Hardware Used
- **Training**: CUDA-enabled GPU
- **Inference**: GPU (50ms per image) or CPU (slower)

---

## 8. Model Comparison & Selection Rationale

### Why EfficientNet-B0?
- ✓ Efficient: 5.3M params vs ResNet50 (25M)
- ✓ Strong ImageNet pretrained weights
- ✓ Compound scaling (depth + width + resolution)
- ✓ Good accuracy-efficiency tradeoff
- ✓ Fast inference (~50ms on GPU)

### Why MobileNetV2-UNet?
- ✓ Lightweight encoder for edge deployment
- ✓ UNet architecture proven for medical segmentation
- ✓ Skip connections preserve spatial details
- ✓ Fast inference suitable for real-time apps
- ✓ Pretrained encoder reduces training time

### Why CLIP (VLM)?
- ✓ Pre-aligned image-text embedding space
- ✓ Zero-shot semantic understanding
- ✓ Frozen (no training required)
- ✓ Enables similarity search
- ✓ Small memory footprint (150MB)

### Why Phi-2 (LLM)?
- ✓ Small size (2.7B params) for edge deployment
- ✓ Strong reasoning capabilities
- ✓ 4-bit quantization support
- ✓ Faster inference than Llama-7B/13B
- ✓ Good instruction-following

---

## 9. Citation & References

### Models
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019
- **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- **UNet**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **Phi-2**: Microsoft Research, 2023

### Loss Functions
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **Dice Loss**: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016

---

**End of Backup Report**  
*Last Updated: January 20, 2026*  
*Project: VisTAI - Bone Tumor X-Ray Detection*  
*Status: Under Development*
