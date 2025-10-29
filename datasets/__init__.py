# datasets/__init__.py
"""
数据集模块
"""

from .infrared_small_target import (
    InfraredSmallTargetDataset, 
    build_infrared_dataset, 
    collate_fn
)
from .rgbt_tiny_config import RGBTinyConfig

__all__ = [
    'InfraredSmallTargetDataset',
    'build_infrared_dataset', 
    'collate_fn',
    'RGBTinyConfig'
]