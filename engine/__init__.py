# engine/__init__.py
"""
训练和评估引擎
"""

from .train import train_one_epoch, evaluate, EarlyStopping
from .evaluate import evaluate_detection, evaluate_model_detailed

__all__ = [
    'train_one_epoch',
    'evaluate', 
    'EarlyStopping',
    'evaluate_detection',
    'evaluate_model_detailed'
]