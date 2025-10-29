# config.py
"""
红外小目标检测统一配置
基于现有的 rgbt_tiny_config.py 构建
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import torch

# 导入现有的数据集配置
from datasets.rgbt_tiny_config import RGBTinyConfig


@dataclass
class ModelConfig:
    """Deformable DETR模型配置 - 内存优化版本"""
    
    # 基础配置
    name: str = "deformable_detr"
    backbone: str = "resnet50"
    hidden_dim: int = 256
    num_queries: int = 50  # 减少默认查询数量
    num_classes: int = 7
    
    # Transformer配置 - 减少默认层数
    nheads: int = 8
    num_encoder_layers: int = 4  # 减少默认层数
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # 训练优化
    auxiliary_loss: bool = False  # 禁用辅助损失以减少内存
    
    # 小目标检测优化
    small_target_enhance: bool = True
    
    def __post_init__(self):
        """配置验证"""
        assert self.backbone in ['resnet34', 'resnet50', 'resnet101'], \
            f"Unsupported backbone: {self.backbone}"
        assert self.hidden_dim % self.nheads == 0, \
            f"hidden_dim must be divisible by nheads: {self.hidden_dim} % {self.nheads} != 0"


@dataclass
class DataConfig:
    """数据加载配置 - 基于现有RGBTinyConfig"""
    
    # 数据集根路径
    dataset_root: str = "datasets/RGBT-Tiny"
    
    # 使用现有的数据集配置实例
    @property
    def dataset(self) -> RGBTinyConfig:
        """获取数据集配置实例"""
        return RGBTinyConfig(root_path=self.dataset_root)
    
    # 数据加载配置
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    
    # 数据增强配置
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_prob: float = 0.3
    
    # 小目标数据增强
    small_target_augmentation: bool = True
    min_scale: float = 0.8
    max_scale: float = 1.2
    
    # 图像预处理
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 标注文件配置
    train_annotations: List[str] = field(default_factory=lambda: [
        "instances_00_train2017.json",
        "instances_01_train2017.json"
    ])
    val_annotations: List[str] = field(default_factory=lambda: [
        "instances_00_test2017.json", 
        "instances_01_test2017.json"
    ])


@dataclass
class TrainConfig:
    """训练配置"""
    
    # 优化器配置
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # 训练调度
    epochs: int = 50
    warmup_epochs: int = 5
    clip_max_norm: float = 0.1
    
    # 损失权重 (针对小目标优化)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'loss_ce': 1.0,      # 分类损失
        'loss_bbox': 5.0,    # 边界框损失 (提高权重)
        'loss_giou': 2.0,    # GIoU损失
    })
    
    # 训练策略
    use_amp: bool = True  # 自动混合精度
    gradient_accumulation: int = 1
    eval_interval: int = 5
    save_interval: int = 10


@dataclass
class ExperimentConfig:
    """实验配置"""
    
    # 实验标识
    experiment_name: str = "infrared_small_target_detection"
    description: str = "基于Deformable DETR的红外小目标检测"
    
    # 路径配置
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # 分布式训练
    distributed: bool = False
    world_size: int = 1
    dist_url: str = "env://"
    
    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.checkpoint_dir), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.log_dir), exist_ok=True)


@dataclass
class Config:
    """总配置类"""
    
    # 实验配置
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 数据配置
    data: DataConfig = field(default_factory=DataConfig)
    
    # 训练配置
    train: TrainConfig = field(default_factory=TrainConfig)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def print_config(config: Config):
    """打印配置信息"""
    print("=" * 60)
    print("红外小目标检测配置")
    print("=" * 60)
    
    print(f"\n📁 实验配置:")
    print(f"  名称: {config.experiment.experiment_name}")
    print(f"  设备: {config.experiment.device}")
    print(f"  输出目录: {config.experiment.output_dir}")
    
    print(f"\n🤖 模型配置:")
    print(f"  骨干网络: {config.model.backbone}")
    print(f"  隐藏维度: {config.model.hidden_dim}")
    print(f"  查询数量: {config.model.num_queries}")
    print(f"  类别数: {config.model.num_classes}")
    print(f"  小目标增强: {config.model.small_target_enhance}")
    
    print(f"\n📊 数据配置:")
    print(f"  数据集根路径: {config.data.dataset_root}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  数据增强: {config.data.use_augmentation}")
    
    # 检查数据集路径
    dataset_cfg = config.data.dataset
    print(f"  图像路径: {dataset_cfg.image_path}")
    print(f"  标注路径: {dataset_cfg.annotation_path}")
    
    print(f"\n🚀 训练配置:")
    print(f"  学习率: {config.train.lr}")
    print(f"  训练轮数: {config.train.epochs}")
    print(f"  权重衰减: {config.train.weight_decay}")
    
    print("=" * 60)


# 预定义配置
def get_resnet50_config() -> Config:
    """ResNet50配置"""
    config = get_default_config()
    config.model.backbone = "resnet50"
    config.model.hidden_dim = 256
    return config


def get_resnet34_config() -> Config:
    """ResNet34配置 - 更轻量"""
    config = get_default_config()
    config.model.backbone = "resnet34"
    config.model.hidden_dim = 128
    config.data.batch_size = 8  # 可以更大的batch_size
    return config


def get_small_target_enhanced_config() -> Config:
    """小目标增强配置"""
    config = get_default_config()
    config.model.small_target_enhance = True
    config.model.use_auxiliary_loss = True
    config.model.num_queries = 150  # 更多查询以检测小目标
    config.train.loss_weights['loss_bbox'] = 8.0  # 提高边界框损失权重
    return config