"""
Fine-tuning module for IMU2CLIP

This package contains:
- text_alternatives.py: Text descriptions for data augmentation
- dataset.py: IMU-Text dataset loader with augmentation
- config.yaml: Default configuration parameters
- finetune.py: Main fine-tuning script
"""

from .text_alternatives import TEXT_ALTERNATIVES
from .dataset import IMUTextDataset, augment_imu_data

__all__ = ['TEXT_ALTERNATIVES', 'IMUTextDataset', 'augment_imu_data']

