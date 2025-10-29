# models/backbone_config.py
"""
红外小目标检测骨干网络配置
针对红外图像特点和小目标检测需求进行优化
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class BackboneConfig:
    """骨干网络配置"""
    # 基础配置
    name: str = "resnet50"  # resnet50, resnet34, resnet101
    pretrained: bool = True
    train_backbone: bool = True
    
    # 红外图像特殊处理
    infrared_adaptation: bool = True  # 是否进行红外图像适配
    use_attention: bool = True  # 是否使用注意力机制增强小目标特征
    
    # 特征金字塔配置
    return_layers: Tuple[str] = ('layer2', 'layer3', 'layer4')  # 返回的中间层
    in_channels_list: List[int] = None  # 各层输出通道数，自动根据backbone设置
    
    # 小目标优化
    feature_enhance: bool = True  # 特征增强
    
    def __post_init__(self):
        """根据backbone类型自动设置通道数"""
        if self.in_channels_list is None:
            if self.name == 'resnet34':
                self.in_channels_list = [128, 256, 512]
            elif self.name == 'resnet50':
                self.in_channels_list = [512, 1024, 2048]
            elif self.name == 'resnet101':
                self.in_channels_list = [512, 1024, 2048]
            else:
                raise ValueError(f"Unsupported backbone: {self.name}")

# 预定义配置
BACKBONE_CONFIGS = {
    'resnet50': BackboneConfig(name='resnet50'),
    'resnet34': BackboneConfig(name='resnet34'),
    'resnet101': BackboneConfig(name='resnet101'),
    'resnet50_enhanced': BackboneConfig(
        name='resnet50',
        infrared_adaptation=True,
        use_attention=True,
        feature_enhance=True
    )
}