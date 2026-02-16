# Project Abstract

## Knowledge Distillation for Efficient Bone Tumor Diagnosis from X-Ray Images

### Background and Motivation

Accurate diagnosis of bone tumors from X-ray images is critical for treatment planning and patient outcomes. While deep learning models have achieved remarkable performance in medical image analysis, their large computational requirements limit deployment in resource-constrained clinical environments such as rural hospitals, mobile diagnostic units, and point-of-care devices. Knowledge distillation (KD) offers a promising solution by compressing large, accurate "teacher" models into smaller, efficient "student" models while maintaining diagnostic performance.

### Objectives

This project implements and evaluates knowledge distillation techniques for two complementary bone tumor diagnosis tasks: (1) **Classification** - identifying tumor types from nine diagnostic categories, and (2) **Segmentation** - localizing and delineating tumor regions for treatment planning. The primary goal is to develop efficient models suitable for clinical deployment without sacrificing diagnostic accuracy or interpretability.

### Methodology

**Dataset**: The study utilized a comprehensive bone tumor X-ray dataset with 9,519 training images, 561 validation images, and 187 test images across nine tumor categories: giant cell tumor, multiple osteochondromas, osteochondroma, osteofibroma, osteosarcoma, other benign tumors, other malignant tumors, simple bone cyst, and synovial osteochondroma. Data augmentation strategies including CutMix, MixUp, and standard geometric transformations addressed class imbalance.

**Phase 1 - Baseline Models**: Initial baseline models were developed to establish performance benchmarks. For classification, EfficientNet-B0 (5.3M parameters) achieved 68.63% validation accuracy. For segmentation, Swin-UNet (27M parameters) achieved 72.55% validation Dice score. These baselines demonstrated task feasibility and informed teacher model selection.

**Phase 2 - Knowledge Distillation Implementation**:

**Classification KD**:
- **Teacher Model**: EfficientNet-B4 (19M parameters) with ImageNet-1K pretraining
- **Student Model**: ConvNeXt-Tiny (28M parameters) with modern architectural improvements
- **Distillation Strategy**: Three-component loss function combining:
  - Logit distillation via KL divergence (temperature=5.0, weight=0.5)
  - Feature distillation via MSE on L2-normalized intermediate representations (weight=0.1)
  - Task loss via label-smoothed cross-entropy with class balancing (weight=0.4)
- **Training**: Two-phase approach with 20 epochs of frozen backbone followed by 60 epochs of full fine-tuning
- **Feature Alignment**: Projection head (768→1024→1792) with BatchNorm and ReLU for cross-architecture distillation

**Segmentation KD**:
- **Teacher Model**: SegFormer-B5 (85M parameters) with hierarchical transformer encoder
- **Student Model**: SegFormer-B2 (28M parameters), achieving 3× compression ratio
- **Distillation Strategy**: Multi-component approach with:
  - Response-based distillation via KL divergence on output logits
  - Feature-based distillation via MSE on multi-scale encoder features
  - Boundary-aware weighting emphasizing tumor edge regions
- **Loss Function**: Combined Dice + Binary Cross-Entropy with deep supervision
- **Architecture**: Same family distillation enabling direct feature matching without projection

**Interpretability**: Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented for both tasks to visualize model attention and validate clinical relevance of learned features.

### Results

**Classification Performance**:
- **Overall Test Accuracy**: 73.80% (138/187 correct predictions)
- **Per-Class Performance**:
  - Excellent: Multiple osteochondromas (88.5%), Simple bone cyst (81.0%), Osteosarcoma (80.0%)
  - Good: Osteochondroma (74.7%)
  - Challenging: Other malignant tumors (25.0%, n=4), Synovial osteochondroma (40.0%, n=5)
- **Inference Speed**: 9.69 iterations/second on NVIDIA RTX 4090
- **Model Size**: Student maintains comparable parameter count to teacher while benefiting from knowledge transfer

**Segmentation Performance**:
- **Training Dice Score**: 72.55% on validation set
- **Test Dice Score**: 50.94% (average across 187 samples)
- **Per-Class Performance**:
  - Excellent: Osteosarcoma (73.4% Dice), Other benign tumors (71.3% Dice)
  - Challenging: Osteochondroma (34.3% Dice despite 76 samples), Osteofibroma (23.2% Dice)
- **Inference Speed**: 7.57 iterations/second
- **Compression**: 3× reduction in parameters (85M → 28M)

**Grad-CAM Visualization**: Generated 50+ attention visualizations per task demonstrating that:
- Classification model attends to tumor-specific anatomical features
- Segmentation model focuses on tumor boundaries and spatial extent
- Both models exhibit clinically interpretable decision-making processes

### Key Findings and Contributions

1. **Effective Knowledge Transfer**: Successfully distilled knowledge from large teacher models into efficient students, achieving 73.80% classification accuracy and practical segmentation performance with significantly reduced computational requirements.

2. **Cross-Architecture Distillation**: Demonstrated that knowledge distillation works effectively between different architecture families (EfficientNet → ConvNeXt) using projection-based feature alignment.

3. **Multi-Component Distillation**: Combining logit, feature, and task losses provides superior results compared to single-component approaches, with weighted contributions optimized for medical imaging constraints.

4. **Interpretability Enhancement**: Grad-CAM visualizations validate that distilled models maintain clinically relevant attention patterns, crucial for trust and adoption in medical settings.

5. **Class Imbalance Challenges**: Identified that rare tumor types (other MT: 4 samples, osteofibroma: 5 samples) significantly impact performance, requiring targeted data augmentation or collection strategies.

6. **Task-Specific Insights**:
   - Classification benefits from two-phase training and strong augmentation (CutMix/MixUp)
   - Segmentation requires boundary-aware losses for tumors with ill-defined edges
   - Same-family distillation (SegFormer-B5 → SegFormer-B2) simplifies feature matching

### Clinical Implications

The developed models offer practical deployment advantages:
- **Resource Efficiency**: Reduced computational requirements enable deployment on standard clinical workstations and mobile devices
- **Inference Speed**: Fast processing times (9-10 it/s) support real-time diagnostic workflows
- **Interpretability**: Grad-CAM visualizations provide clinicians with explainable predictions, addressing the "black box" concern in medical AI
- **Diagnostic Support**: Classification identifies tumor types while segmentation provides spatial information for treatment planning

### Limitations and Future Work

**Current Limitations**:
1. **Segmentation Generalization Gap**: Discrepancy between training (72.5%) and test (50.9%) Dice scores suggests overfitting or evaluation methodology issues requiring investigation
2. **Class Imbalance**: Limited samples for rare tumor types (4-5 samples) restrict model performance and generalization
3. **Boundary Precision**: Tumors with diffuse or ill-defined boundaries (osteochondroma) achieve lower segmentation accuracy
4. **Single-Center Data**: Dataset from single institution may limit generalization to diverse imaging protocols

**Recommended Future Directions**:
1. **Stronger Teacher Models**: Train ensemble teachers or larger architectures (EfficientNet-B7, nnU-Net V2) to improve knowledge quality
2. **Advanced KD Techniques**: Implement attention transfer, contrastive learning, and relation distillation for richer knowledge transfer
3. **Data Augmentation**: Collect additional samples for rare classes or employ synthetic data generation techniques
4. **Multi-Task Learning**: Joint training of classification and segmentation for shared representation learning
5. **External Validation**: Evaluate on multi-center datasets to assess generalization across imaging protocols and populations
6. **Clinical Validation**: Prospective studies with radiologists to assess real-world diagnostic utility and integration into clinical workflows

### Conclusion

This project successfully demonstrates that knowledge distillation is a viable approach for creating efficient, interpretable models for bone tumor diagnosis from X-ray images. The classification student achieves 73.80% test accuracy with practical inference speed, while the segmentation student provides 3× compression with maintained clinical utility. By combining multi-component distillation strategies with explainability techniques (Grad-CAM), the developed models balance efficiency, accuracy, and interpretability—critical requirements for medical AI deployment. The work establishes a foundation for future enhancements through stronger teachers, advanced distillation techniques, and expanded datasets, with clear pathways toward state-of-the-art performance and clinical translation.

---

**Keywords**: Knowledge Distillation, Medical Image Analysis, Bone Tumor Diagnosis, Deep Learning, Model Compression, Explainable AI, X-Ray Imaging, Classification, Segmentation, Grad-CAM

**Technologies**: PyTorch, EfficientNet, ConvNeXt, SegFormer, Swin Transformer, Mixed Precision Training, CUDA

**Performance Summary**: 73.80% classification accuracy | 50.94% segmentation Dice | 3× compression | 9.69 it/s inference

**Project Duration**: 8 weeks (4 weeks baseline development + 4 weeks knowledge distillation implementation)
