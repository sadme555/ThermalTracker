"""
配置文件管理
使用yaml文件管理所有配置参数
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """数据配置"""
    dataset_root: str = "datasets/RGBT-Tiny"
    num_classes: int = 1
    image_size: List[int] = field(default_factory=lambda: [512, 640])
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    small_target_augmentation: bool = True
    min_scale: float = 0.8
    max_scale: float = 1.2
    color_jitter_prob: float = 0.5
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

@dataclass
class ModelConfig:
    """模型配置"""
    backbone: str = "resnet50"
    hidden_dim: int = 256
    num_queries: int = 100
    nheads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_feature_levels: int = 3

@dataclass
class LossConfig:
    """损失配置"""
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    weight_class: float = 1.0
    weight_bbox: float = 5.0
    weight_giou: float = 2.0
    eos_coef: float = 0.1

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    device: str = "cuda"
    seed: int = 42
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    
    # 优化器配置
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    
    # 学习率调度
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # 检查点和保存
    output_dir: str = "outputs"
    resume: str = ""
    checkpoint_interval: int = 10
    
    # 早停
    patience: int = 20
    min_delta: float = 1e-4

@dataclass
class EvaluationConfig:
    """评估配置"""
    confidence_threshold: float = 0.5
    checkpoint_path: str = "outputs/best_model.pth"

@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置"""
        config = cls()
        
        # 递归更新配置
        for key, value in config_dict.items():
            if hasattr(config, key):
                sub_config = getattr(config, key)
                if hasattr(sub_config, '__dataclass_fields__'):
                    # 更新子配置
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(config, key, value)
        
        return config

def setup_device():
    """根据CUDA可用性设置设备"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def get_config(config_path: str = None) -> Config:
    """加载配置文件"""
    # 首先设置设备
    device = setup_device()
    
    if config_path is None or not os.path.exists(config_path):
        # 返回默认配置
        print(f"Using default config, config file not found: {config_path}")
        config = Config()
        config.training.device = device
        return config
    
    # 尝试不同的编码方式读取配置文件
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is not None:
                config = Config.from_dict(config_dict)
                config.training.device = device
                print(f"Successfully loaded config with {encoding} encoding")
                return config
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            print(f"Failed to load config with {encoding} encoding: {e}")
            continue
    
    # 如果所有编码都失败，使用默认配置
    print(f"All encoding attempts failed for {config_path}, using default config")
    config = Config()
    config.training.device = device
    return config

def save_config(config: Config, config_path: str):
    """保存配置到文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config_dict = {}
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if hasattr(value, '__dataclass_fields__'):
            # 处理子配置
            sub_dict = {}
            for sub_field in value.__dataclass_fields__:
                sub_dict[sub_field] = getattr(value, sub_field)
            config_dict[field] = sub_dict
        else:
            config_dict[field] = value
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Config saved to: {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")

def create_default_config():
    """创建默认配置文件"""
    default_config = Config()
    default_config.training.device = setup_device()
    
    os.makedirs('configs', exist_ok=True)
    save_config(default_config, 'configs/default.yaml')
    print("Default config created at: configs/default.yaml")

if __name__ == '__main__':
    create_default_config()