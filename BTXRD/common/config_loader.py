"""Configuration file loader."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class Config:
    """Configuration object with dot notation access."""
    
    def __init__(self, config_dict):
        """
        Args:
            config_dict: Dictionary with configuration
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


def load_config_object(config_path):
    """
    Load configuration as object with dot notation access.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Config object
    """
    config_dict = load_config(config_path)
    return Config(config_dict)


if __name__ == "__main__":
    # Test config loader
    print("Testing config loader...")
    
    # Example usage
    test_config = {
        'model': {'name': 'efficientnet', 'num_classes': 9},
        'training': {'batch_size': 32, 'lr': 0.001}
    }
    
    config = Config(test_config)
    
    print(f"Model name: {config.model.name}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"\nConfig dict: {config.to_dict()}")
    
    print("\n✓ Config loader test passed!")
