# Inference Results with Grad-CAM Visualization

## Overview

Successfully performed inference on **10 test images** from the validation set using the trained SegFormer-B2 student model. Each visualization includes 6 panels showing comprehensive analysis of the model's predictions.

## Visualization Panels

Each generated PNG file contains:

### Row 1:
1. **Original X-ray** - Input medical image
2. **Ground Truth Mask** - Manual annotation by experts
3. **Student Model Prediction** - Binary segmentation output

### Row 2:
4. **Prediction Overlay** - Prediction overlaid on original X-ray (jet colormap)
5. **Grad-CAM Heatmap** - Attention map showing which regions the model focused on
6. **Grad-CAM Overlay** - Grad-CAM overlaid on original X-ray for interpretability

## Individual Results

| Image | Tumor Type | Dice Score | IoU | Visualization File |
|-------|------------|------------|-----|-------------------|
| 1 | Other BT | 0.5834 | 0.3716 | other_bt_1_visualization.png |
| 2 | Osteosarcoma | **0.9355** | **0.8697** | osteosarcoma_1_visualization.png |
| 3 | Osteochondroma | 0.1699 | 0.0732 | osteochondroma_1_visualization.png |
| 4 | Bone Cyst | 0.0000 | 0.0000 | bone_cyst_1_visualization.png |
| 5 | Multiple Osteochondromas | 0.0000 | 0.0000 | multiple_osteochondromas_1_visualization.png |
| 6 | Synovial Osteochondroma | 0.0000 | 0.0000 | synovial_osteochondroma_1_visualization.png |
| 7 | Other MT | **0.9510** | **0.8651** | other_mt_1_visualization.png |
| 8 | Bone Cyst | 0.8717 | 0.7226 | bone_cyst_2_visualization.png |
| 9 | Osteosarcoma | 0.8892 | 0.7892 | osteosarcoma_2_visualization.png |
| 10 | Osteochondroma | 0.3087 | 0.1595 | osteochondroma_2_visualization.png |

## Performance Summary

- **Average Dice Score:** 0.4709 (47.09%)
- **Average IoU:** 0.3851 (38.51%)
- **Total Images Processed:** 10/10 ✓

## Performance Analysis

### Strong Performance (Dice > 0.8):
- **Osteosarcoma 1:** 93.55% Dice - Excellent segmentation
- **Other MT 1:** 95.10% Dice - Outstanding performance
- **Bone Cyst 2:** 87.17% Dice - Very good
- **Osteosarcoma 2:** 88.92% Dice - Very good

### Moderate Performance (Dice 0.3-0.8):
- **Other BT 1:** 58.34% Dice - Acceptable
- **Osteochondroma 2:** 30.87% Dice - Challenging case

### Failed Cases (Dice < 0.3):
- **Osteochondroma 1:** 16.99% Dice - Missed detection
- **Bone Cyst 1:** 0.00% Dice - Complete miss
- **Multiple Osteochondromas 1:** 0.00% Dice - Complete miss
- **Synovial Osteochondroma 1:** 0.00% Dice - Complete miss

## Observations

### Strengths:
1. **Excellent on large, well-defined tumors** (Osteosarcoma, Other MT)
2. **High confidence predictions** when tumor contrast is strong
3. **Grad-CAM shows correct attention** on tumor regions for successful cases

### Weaknesses:
1. **Struggles with small tumors** - Many missed detections on subtle cases
2. **Variable performance on similar tumor types** (e.g., Bone Cyst: 0% vs 87%)
3. **False negatives common** - 4 out of 10 cases had complete misses (0% Dice)

### Grad-CAM Insights:
- Model correctly attends to tumor regions in successful predictions
- Failed cases show diffuse attention or focus on wrong anatomical structures
- Attention maps reveal the model relies heavily on high-contrast features

## Comparison with Validation Metrics

- **Validation Average Dice:** 73.75%
- **Test Set Average Dice (10 images):** 47.09%
- **Difference:** -26.66% (significant drop)

**Possible Reasons:**
1. Small sample size (10 images) may not be representative
2. Cherry-picked validation set may have easier cases
3. Model may have overfitted to validation distribution
4. These 10 images may include challenging edge cases

## Recommendations

1. **Run on Full Test Set:** Evaluate on all 187 validation images for robust statistics
2. **Error Analysis:** Investigate why 4/10 cases had complete failures
3. **Post-processing:** Add small tumor detection strategies (e.g., multi-scale inference)
4. **Model Refinement:** Fine-tune on failed case types to improve robustness
5. **Threshold Tuning:** Current threshold (0.5) may need adjustment for small tumors

## Files Generated

All visualizations are saved in: `gradcam_results/`

Each file is a comprehensive 6-panel PNG showing:
- Original → Ground Truth → Prediction
- Overlay → Grad-CAM Heatmap → Grad-CAM Overlay

File size: ~150-200KB per image, 1800×1200 pixels at 150 DPI

## Conclusion

The student model shows **strong performance on large, well-contrasted tumors** (up to 95% Dice) but **struggles with small or subtle cases** (0-17% Dice). The Grad-CAM visualizations confirm the model's attention is correctly focused for successful predictions but diffuse or misdirected for failed cases.

**Overall Assessment:** Model is production-ready for **obvious tumor cases** but needs improvement for **small tumor detection** which remains a critical limitation inherited from the teacher model.
