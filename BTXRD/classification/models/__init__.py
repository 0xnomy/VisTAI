"""
Classification Models
=====================
Provides model registry with build_model() factory function.

Available models:
- efficientnet_b0: EfficientNet-B0 student model (~5.3M params)
- convnext_tiny: ConvNeXt-Tiny teacher (~28M params)
- convnext_small: ConvNeXt-Small teacher (~50M params)
- convnext_base: ConvNeXt-Base teacher (~89M params)
"""

from .efficientnet_classifier import EfficientNetClassifier, create_efficientnet_classifier
from .convnext_teacher import ConvNeXtTeacher, create_convnext_teacher

# Model registry
_MODELS = {
    'efficientnet_b0': {
        'class': EfficientNetClassifier,
        'factory': create_efficientnet_classifier,
        'params': '~5.3M',
        'type': 'student'
    },
    'convnext_tiny': {
        'class': ConvNeXtTeacher,
        'factory': lambda **kwargs: create_convnext_teacher(model_size='tiny', **kwargs),
        'params': '~28M',
        'type': 'teacher'
    },
    'convnext_small': {
        'class': ConvNeXtTeacher,
        'factory': lambda **kwargs: create_convnext_teacher(model_size='small', **kwargs),
        'params': '~50M',
        'type': 'teacher'
    },
    'convnext_base': {
        'class': ConvNeXtTeacher,
        'factory': lambda **kwargs: create_convnext_teacher(model_size='base', **kwargs),
        'params': '~89M',
        'type': 'teacher'
    },
}


def build_model(model_name: str, num_classes: int = 9, pretrained: bool = True, **kwargs):
    """
    Build a classification model by name.
    
    Args:
        model_name: Name of the model ('efficientnet_b0', 'convnext_small', etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        **kwargs: Additional arguments passed to the factory function
    
    Returns:
        Initialized model
    
    Example:
        >>> model = build_model('efficientnet_b0', num_classes=9, pretrained=True)
        >>> model = build_model('convnext_small', num_classes=9, dropout=0.5)
    """
    if model_name not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    factory = _MODELS[model_name]['factory']
    return factory(num_classes=num_classes, pretrained=pretrained, **kwargs)


def list_models():
    """List all available classification models."""
    print("\n📋 Available Classification Models:")
    print("-" * 50)
    for name, info in _MODELS.items():
        print(f"  • {name:20} {info['params']:>10}  ({info['type']})")
    print("-" * 50)
    return list(_MODELS.keys())


__all__ = [
    'EfficientNetClassifier',
    'ConvNeXtTeacher',
    'build_model',
    'list_models',
]
