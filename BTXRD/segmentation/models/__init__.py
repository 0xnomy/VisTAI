"""
Segmentation Models
===================
Provides model registry with build_model() factory function.

Available models:
- nnunet: nnU-Net-style student model (~7.8M params)
- swin_unet: Swin-UNet transformer model (~50M params, SOTA teacher)
- unetplusplus_resnet50: UNet++ with ResNet50 encoder (~32M params)
- unetplusplus_efficientnet_b4: UNet++ with EfficientNet-B4 encoder (~20M params)
- deeplabv3plus_resnet50: DeepLabV3+ with ResNet50 encoder (~26M params)
- manet_efficientnet_b4: MAnet with EfficientNet-B4 encoder (~21M params)
"""

from .nnunet_student import NNUNetStudent, create_nnunet_student
from .teachers import (
    UNetPlusPlusTeacher, DeepLabV3PlusTeacher, MAnetTeacher,
    create_unetplusplus_teacher, create_deeplabv3plus_teacher, create_manet_teacher
)
from .swin_unet import SwinUNet, create_swin_unet

# Model registry
_MODELS = {
    # Student model
    'nnunet': {
        'class': NNUNetStudent,
        'factory': create_nnunet_student,
        'params': '~7.8M',
        'type': 'student'
    },
    
    # Swin-UNet (State-of-the-art transformer teacher)
    'swin_unet': {
        'class': SwinUNet,
        'factory': create_swin_unet,
        'params': '~50M',
        'type': 'teacher'
    },
    
    # UNet++ variants
    'unetplusplus_resnet50': {
        'class': UNetPlusPlusTeacher,
        'factory': lambda **kwargs: create_unetplusplus_teacher(encoder_name='resnet50', **kwargs),
        'params': '~32M',
        'type': 'teacher'
    },
    'unetplusplus_resnet101': {
        'class': UNetPlusPlusTeacher,
        'factory': lambda **kwargs: create_unetplusplus_teacher(encoder_name='resnet101', **kwargs),
        'params': '~51M',
        'type': 'teacher'
    },
    'unetplusplus_efficientnet_b3': {
        'class': UNetPlusPlusTeacher,
        'factory': lambda **kwargs: create_unetplusplus_teacher(encoder_name='efficientnet-b3', **kwargs),
        'params': '~15M',
        'type': 'teacher'
    },
    'unetplusplus_efficientnet_b4': {
        'class': UNetPlusPlusTeacher,
        'factory': lambda **kwargs: create_unetplusplus_teacher(encoder_name='efficientnet-b4', **kwargs),
        'params': '~20M',
        'type': 'teacher'
    },
    
    # DeepLabV3+ variants
    'deeplabv3plus_resnet50': {
        'class': DeepLabV3PlusTeacher,
        'factory': lambda **kwargs: create_deeplabv3plus_teacher(encoder_name='resnet50', **kwargs),
        'params': '~26M',
        'type': 'teacher'
    },
    'deeplabv3plus_resnet101': {
        'class': DeepLabV3PlusTeacher,
        'factory': lambda **kwargs: create_deeplabv3plus_teacher(encoder_name='resnet101', **kwargs),
        'params': '~45M',
        'type': 'teacher'
    },
    
    # MAnet variants
    'manet_efficientnet_b4': {
        'class': MAnetTeacher,
        'factory': lambda **kwargs: create_manet_teacher(encoder_name='efficientnet-b4', **kwargs),
        'params': '~21M',
        'type': 'teacher'
    },
    'manet_resnet50': {
        'class': MAnetTeacher,
        'factory': lambda **kwargs: create_manet_teacher(encoder_name='resnet50', **kwargs),
        'params': '~32M',
        'type': 'teacher'
    },
}


def build_model(model_name: str, in_channels: int = 3, classes: int = 1, pretrained: bool = True, **kwargs):
    """
    Build a segmentation model by name.
    
    Args:
        model_name: Name of the model ('nnunet', 'unetplusplus_resnet50', etc.)
        in_channels: Number of input channels (default: 3 for RGB)
        classes: Number of output classes (default: 1 for binary segmentation)
        pretrained: Use pretrained encoder weights (for teacher models)
        **kwargs: Additional arguments passed to the factory function
    
    Returns:
        Initialized model
    
    Example:
        >>> model = build_model('nnunet', deep_supervision=True)
        >>> model = build_model('unetplusplus_resnet50', pretrained=True)
    """
    if model_name not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    factory = _MODELS[model_name]['factory']
    
    # Handle different factory signatures
    if model_name == 'nnunet':
        return factory(in_channels=in_channels, out_channels=classes, **kwargs)
    elif model_name == 'swin_unet':
        return factory(num_classes=classes, pretrained=pretrained, **kwargs)
    else:
        encoder_weights = 'imagenet' if pretrained else None
        return factory(in_channels=in_channels, classes=classes, encoder_weights=encoder_weights, **kwargs)


def list_models():
    """List all available segmentation models."""
    print("\n📋 Available Segmentation Models:")
    print("-" * 60)
    for name, info in _MODELS.items():
        print(f"  • {name:35} {info['params']:>10}  ({info['type']})")
    print("-" * 60)
    return list(_MODELS.keys())


__all__ = [
    'NNUNetStudent',
    'SwinUNet',
    'UNetPlusPlusTeacher',
    'DeepLabV3PlusTeacher',
    'MAnetTeacher',
    'build_model',
    'list_models',
]
