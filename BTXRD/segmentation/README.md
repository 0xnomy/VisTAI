# Bone Tumor Segmentation

Binary tumor segmentation using nnU-Net-style student and UNet++/DeepLabV3+/MAnet teacher models.

## Quick Start

```bash
# Train student model (nnU-Net)
python train.py --model nnunet --epochs 100

# Train teacher model (UNet++ with ResNet50)
python train.py --model unetplusplus_resnet50 --epochs 100

# Run inference
python inference.py --model nnunet --checkpoint outputs/nnunet/checkpoint_best.pth --visualize
```

## Available Models

| Model | Parameters | Type |
|-------|-----------|------|
| `nnunet` | ~7.8M | Student |
| `unetplusplus_resnet50` | ~32M | Teacher |
| `unetplusplus_resnet101` | ~51M | Teacher |
| `unetplusplus_efficientnet_b4` | ~20M | Teacher |
| `deeplabv3plus_resnet50` | ~26M | Teacher |
| `deeplabv3plus_resnet101` | ~45M | Teacher |
| `manet_efficientnet_b4` | ~21M | Teacher |
| `manet_resnet50` | ~32M | Teacher |

## Directory Structure

```
segmentation/
├── train.py                          # Training script
├── inference.py                      # Inference script
├── models/
│   ├── __init__.py                   # Model registry with build_model()
│   ├── nnunet_student.py             # nnU-Net student model
│   └── teachers.py                   # UNet++, DeepLabV3+, MAnet teachers
├── datasets/
│   ├── __init__.py
│   └── segmentation_dataset.py       # Data loading
├── configs/
│   └── teacher_config.yaml
├── outputs/                          # Checkpoints and history
└── README.md
```

## Usage

### Model Selection via `build_model()`

```python
from segmentation.models import build_model, list_models

# List available models
list_models()

# Build model by name
model = build_model('nnunet', deep_supervision=True)
model = build_model('unetplusplus_resnet50', pretrained=True)
```

### Training Options

```bash
python train.py --model nnunet --epochs 100 --batch-size 8 --lr 0.001
python train.py --model unetplusplus_resnet50 --epochs 100 --image-size 256
```

### Inference Options

```bash
# Single image
python inference.py --model nnunet --checkpoint outputs/nnunet/checkpoint_best.pth --image path/to/image.jpg --mask path/to/mask.png

# Batch from CSV
python inference.py --model nnunet --checkpoint outputs/nnunet/checkpoint_best.pth --csv test.csv --num-samples 10 --visualize
```

## nnU-Net Student Architecture

- **Encoder**: 5 stages with channels [32, 64, 128, 256, 512]
- **Decoder**: 4 upsampling stages with skip connections
- **Normalization**: Instance Normalization
- **Activation**: LeakyReLU (slope=0.01)
- **Deep Supervision**: Auxiliary outputs at 1/2, 1/4, 1/8 resolution
- **Loss**: Dice + BCE combined loss
