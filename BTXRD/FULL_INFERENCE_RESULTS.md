# Full Dataset Inference Results with Grad-CAM

This document summarizes the inference results on the complete test datasets for both classification and segmentation knowledge distillation models.

---

## Classification KD - Full Test Dataset Results

### Dataset
- **Test samples**: 187 images
- **Classes**: 9 bone tumor types
- **Model**: ConvNeXt-Tiny Student (distilled from EfficientNet-B4)
- **Checkpoint**: `outputs/kd_student/best_model.pth`

### Overall Performance
- **Accuracy**: **73.80%** (138/187 correct predictions)
- **Visualizations**: 50 Grad-CAM visualizations generated

### Per-Class Performance

| Tumor Type | Accuracy | Correct | Total | Performance Level |
|-----------|----------|---------|-------|-------------------|
| **Multiple osteochondromas** | **88.5%** | 23 | 26 | ⭐⭐⭐ Excellent |
| **Simple bone cyst** | **81.0%** | 17 | 21 | ⭐⭐⭐ Excellent |
| **Osteofibroma** | **80.0%** | 4 | 5 | ⭐⭐⭐ Excellent |
| **Osteosarcoma** | **80.0%** | 24 | 30 | ⭐⭐⭐ Excellent |
| **Osteochondroma** | **74.7%** | 56 | 75 | ⭐⭐ Good |
| **Giant cell tumor** | **55.6%** | 5 | 9 | ⭐ Fair |
| **Other BT** | **50.0%** | 6 | 12 | ⭐ Fair |
| **Synovial osteochondroma** | **40.0%** | 2 | 5 | ⚠️ Poor |
| **Other MT** | **25.0%** | 1 | 4 | ⚠️ Poor |

### Key Observations

**Strengths:**
- Strong performance on common tumor types (osteochondroma: 74.7%, 75 samples)
- Excellent accuracy on multiple osteochondromas (88.5%)
- High performance on malignant tumors (osteosarcoma: 80.0%)
- Robust results on simple bone cysts (81.0%)

**Challenges:**
- Lower accuracy on rare classes with few samples (other MT: 25%, n=4)
- Moderate performance on synovial osteochondroma (40%, n=5)
- Class imbalance effects visible (osteochondroma: 75 samples vs other MT: 4 samples)

**Class Imbalance:**
- Largest class: Osteochondroma (75 samples, 40.1%)
- Smallest classes: Other MT (4 samples), Osteofibroma (5 samples), Synovial osteochondroma (5 samples)

### Analysis
The model achieves **73.80% overall accuracy** on the full test set, demonstrating effective knowledge distillation from the EfficientNet-B4 teacher. Performance correlates with class size - larger classes (osteochondroma, osteosarcoma) show better accuracy. The model struggles with rare tumor types and morphologically similar variants (synovial vs regular osteochondroma).

---

## Segmentation KD - Full Validation Dataset Results

### Dataset
- **Validation samples**: 187 images with segmentation masks
- **Task**: Binary tumor segmentation (tumor vs background)
- **Model**: SegFormer-B2 Student (distilled from SegFormer-B5)
- **Checkpoint**: `outputs/kd_student/best_model.pth`

### Overall Performance
- **Average Dice Score**: **0.5094** (50.94%)
- **Average IoU**: **0.4256** (42.56%)
- **Visualizations**: 50 Grad-CAM visualizations generated

### Per-Class Performance

| Tumor Type | Dice Score | IoU | Samples | Performance Level |
|-----------|-----------|-----|---------|-------------------|
| **Osteosarcoma** | **0.7341** | **0.6330** | 29 | ⭐⭐⭐ Excellent |
| **Other BT** | **0.7125** | **0.5860** | 11 | ⭐⭐⭐ Excellent |
| **Giant cell tumor** | **0.6977** | **0.6208** | 10 | ⭐⭐ Good |
| **Other MT** | **0.6305** | **0.5348** | 5 | ⭐⭐ Good |
| **Multiple osteochondromas** | **0.6168** | **0.4883** | 27 | ⭐⭐ Good |
| **Simple bone cyst** | **0.5227** | **0.4580** | 20 | ⭐ Fair |
| **Synovial osteochondroma** | **0.3802** | **0.3347** | 5 | ⚠️ Poor |
| **Osteochondroma** | **0.3428** | **0.2775** | 76 | ⚠️ Poor |
| **Osteofibroma** | **0.2322** | **0.1959** | 4 | ⚠️ Poor |

### Key Observations

**Strengths:**
- Excellent segmentation on osteosarcoma (Dice: 0.734, n=29)
- Strong performance on other benign tumors (Dice: 0.713, n=11)
- Good results on giant cell tumors (Dice: 0.698, n=10)

**Challenges:**
- Poor performance on osteochondroma despite large sample size (Dice: 0.343, n=76)
- Low accuracy on osteofibroma (Dice: 0.232, n=4)
- Moderate results on synovial osteochondroma (Dice: 0.380, n=5)

**Class Distribution:**
- Largest class: Osteochondroma (76 samples, 40.6%)
- Well-represented: Osteosarcoma (29 samples), Multiple osteochondromas (27 samples)
- Small classes: Osteofibroma (4 samples), Other MT (5 samples), Synovial osteochondroma (5 samples)

### Analysis
The segmentation model achieves **50.94% average Dice score**, which is lower than the training-reported 73.75%. This discrepancy suggests:
1. **Overfitting**: Model may have overfit to training data
2. **Data distribution**: Validation set may be more challenging
3. **Tumor complexity**: Some tumor types (osteochondroma) have subtle boundaries that are harder to segment

The model excels at segmenting aggressive/malignant tumors (osteosarcoma) and well-defined lesions (other BT, giant cell tumor) but struggles with diffuse or ill-defined boundaries (osteochondroma, osteofibroma).

---

## Comparative Analysis: Classification vs Segmentation

### Model Performance

| Metric | Classification | Segmentation | Winner |
|--------|---------------|--------------|--------|
| Overall accuracy | 73.80% | 50.94% (Dice) | Classification ✓ |
| Best class performance | 88.5% (multiple osteochondromas) | 73.4% (osteosarcoma) | Classification ✓ |
| Worst class performance | 25.0% (other MT) | 23.2% (osteofibroma) | Similar |
| Processing speed | 9.69 it/s | 7.57 it/s | Classification ✓ |

### Inference Statistics

| Aspect | Classification | Segmentation |
|--------|---------------|--------------|
| Dataset size | 187 test images | 187 validation images |
| Processing time | ~19 seconds | ~24 seconds |
| Visualizations | 50 Grad-CAM plots | 50 Grad-CAM plots |
| Output format | Predictions + confidence | Masks + Dice/IoU |

### Task-Specific Insights

**Classification Advantages:**
- Higher overall accuracy (73.80% vs 50.94%)
- Faster inference (9.69 vs 7.57 it/s)
- More robust to class imbalance
- Clearer decision boundaries

**Segmentation Advantages:**
- Provides spatial localization of tumors
- Enables precise boundary delineation
- More clinically actionable (tumor extent)
- Better interpretability with Grad-CAM overlays

**Common Challenges:**
- Both models struggle with rare classes (other MT, osteofibroma)
- Morphologically similar tumors are difficult (synovial vs regular osteochondroma)
- Class imbalance affects performance
- Small sample sizes lead to unstable estimates

---

## Grad-CAM Visualization Quality

### Classification Grad-CAM
- **Target layer**: Last convolutional layer of ConvNeXt-Tiny
- **Visualization**: 4-panel layout (original, heatmap, overlay, predictions)
- **Interpretability**: Highlights discriminative regions for tumor type classification
- **Clinical value**: Shows if model attends to tumor location vs. irrelevant features

### Segmentation Grad-CAM
- **Target layer**: Decoder head convolutional layer
- **Visualization**: 6-panel layout (original, GT mask, prediction, heatmap, overlay, comparison)
- **Interpretability**: Shows attention on tumor boundaries
- **Clinical value**: Validates segmentation quality and boundary precision

---

## Recommendations

### For Classification KD
1. **Address class imbalance**: Use class-balanced sampling or weighted loss
2. **Augment rare classes**: Generate more samples for other MT, synovial osteochondroma
3. **Fine-tune on confusing pairs**: Focus training on osteochondroma variants
4. **Ensemble methods**: Combine multiple students for robust predictions

### For Segmentation KD
1. **Investigate osteochondroma performance**: Large class but poor Dice (0.343)
   - May have subtle/diffuse boundaries
   - Consider boundary-aware loss functions
2. **Reduce overfitting**: Add regularization, increase validation monitoring
3. **Boundary refinement**: Use edge-focused augmentation and loss
4. **Post-processing**: Apply morphological operations to smooth masks

### General Improvements
1. **Collect more data**: Especially for rare classes (other MT, osteofibroma)
2. **Expert annotation review**: Verify ground truth quality for low-performing classes
3. **Cross-validation**: Implement k-fold CV for more robust performance estimates
4. **Multi-task learning**: Combine classification and segmentation objectives

---

## Conclusion

Both knowledge distillation models demonstrate practical utility:
- **Classification KD**: Achieves **73.80% accuracy** with fast inference (9.69 it/s)
- **Segmentation KD**: Achieves **50.94% Dice** with spatial localization

The classification model is **more reliable** for tumor type identification, while the segmentation model provides **spatial information** crucial for treatment planning. A **combined pipeline** leveraging both models could offer comprehensive diagnostic support:
1. **Classification** for tumor type diagnosis
2. **Segmentation** for tumor extent and surgical planning
3. **Grad-CAM** for explainability and trust

The results validate knowledge distillation as an effective compression technique for medical imaging, enabling deployment of capable models in resource-constrained clinical settings.
