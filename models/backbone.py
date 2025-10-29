# models/backbone.py
"""
红外小目标检测骨干网络 - 统一修复版本
基于ResNet，完全控制各层以避免通道不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List

from .backbone_config import BackboneConfig


class InfraredAdaptation(nn.Module):
    """
    红外图像适配模块
    替代ResNet的早期层，专门为红外图像优化
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.adapter(x)


class Backbone(nn.Module):
    """
    统一的骨干网络实现
    完全控制ResNet各层，避免IntermediateLayerGetter的问题
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练ResNet
        if config.name == 'resnet50':
            resnet = models.resnet50(pretrained=config.pretrained)
            self.channel_list = [512, 1024, 2048]  # layer2, layer3, layer4的输出通道
        elif config.name == 'resnet34':
            resnet = models.resnet34(pretrained=config.pretrained)
            self.channel_list = [128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {config.name}")
        
        # 红外适配
        if config.infrared_adaptation:
            self.infra_adapt = InfraredAdaptation(3, 64)
        else:
            # 使用ResNet原生的早期层
            self.infra_adapt = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        
        # ResNet的主体层
        self.layer1 = resnet.layer1  # 输出: 256 (resnet50) 或 64 (resnet34)
        self.layer2 = resnet.layer2  # 输出: 512 (resnet50) 或 128 (resnet34)  
        self.layer3 = resnet.layer3  # 输出: 1024 (resnet50) 或 256 (resnet34)
        self.layer4 = resnet.layer4  # 输出: 2048 (resnet50) 或 512 (resnet34)
        
        # 特征金字塔增强
        self.feature_enhance = config.feature_enhance
        if config.feature_enhance:
            self.fpn = FeaturePyramidNetwork(self.channel_list)
        
        # 冻结早期层（可选）
        if not config.train_backbone:
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        前向传播
        返回多尺度特征字典
        """
        # 红外适配
        x = self.infra_adapt(x)  # [batch, 64, H/4, W/4]
        
        # 通过ResNet各层
        c2 = self.layer1(x)  # [batch, 256/64, H/4, W/4]
        c3 = self.layer2(c2) # [batch, 512/128, H/8, W/8]  
        c4 = self.layer3(c3) # [batch, 1024/256, H/16, W/16]
        c5 = self.layer4(c4) # [batch, 2048/512, H/32, W/32]
        
        # 构建特征字典
        features = {'0': c3, '1': c4, '2': c5}
        
        # 特征金字塔增强
        if self.feature_enhance:
            features = self.fpn(features)
        
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络
    用于多尺度特征融合，特别适合小目标检测
    """
    
    def __init__(self, channel_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # 横向连接（1x1卷积降维）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channel, out_channels, 1) 
            for channel in channel_list
        ])
        
        # 融合卷积（3x3卷积细化特征）
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in channel_list
        ])

    def forward(self, features):
        """
        输入: 特征字典 {'0': c3, '1': c4, '2': c5}
        输出: 增强后的特征字典，所有特征图通道数相同
        """
        # 提取各层特征
        c3, c4, c5 = features['0'], features['1'], features['2']
        
        # 自上而下的特征融合
        p5 = self.lateral_convs[2](c5)  # 最顶层
        
        # p4 = c4 + 上采样的p5
        p4 = self.lateral_convs[1](c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode='nearest'
        )
        
        # p3 = c3 + 上采样的p4  
        p3 = self.lateral_convs[0](c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode='nearest'
        )
        
        # 融合卷积细化特征
        p3 = self.fusion_convs[0](p3)
        p4 = self.fusion_convs[1](p4)
        p5 = self.fusion_convs[2](p5)
        
        return {'0': p3, '1': p4, '2': p5}


def build_backbone(config: BackboneConfig):
    """构建骨干网络"""
    return Backbone(config)


# 测试函数
def test_backbone():
    """测试骨干网络"""
    print("=" * 60)
    print("Testing Unified Backbone")
    print("=" * 60)
    
    # 测试不同配置
    test_configs = [
        {
            'name': 'resnet50', 
            'infrared_adaptation': False,
            'feature_enhance': False,
            'description': 'ResNet50 - 无红外适配'
        },
        {
            'name': 'resnet50', 
            'infrared_adaptation': True,
            'feature_enhance': False, 
            'description': 'ResNet50 - 有红外适配'
        },
        {
            'name': 'resnet50',
            'infrared_adaptation': True,
            'feature_enhance': True,
            'description': 'ResNet50 - 完整配置'
        },
    ]
    
    for i, cfg in enumerate(test_configs):
        print(f"\nTest {i+1}: {cfg['description']}")
        
        try:
            # 创建配置
            config = BackboneConfig(
                name=cfg['name'],
                pretrained=False,  # 测试时不用预训练权重
                infrared_adaptation=cfg['infrared_adaptation'],
                feature_enhance=cfg['feature_enhance']
            )
            
            # 构建骨干网络
            backbone = build_backbone(config)
            print("✓ Backbone created successfully")
            
            # 测试输入
            x = torch.randn(2, 3, 512, 640)
            print(f"Input shape: {x.shape}")
            
            # 前向传播
            with torch.no_grad():
                features = backbone(x)
            
            print("✓ Forward pass successful")
            print(f"Feature levels: {len(features)}")
            
            # 检查输出特征
            for name, feature in features.items():
                print(f"  {name}: {feature.shape}")
                
                # 验证特征图尺寸
                expected_channels = 256 if cfg['feature_enhance'] else config.in_channels_list[int(name)]
                actual_channels = feature.shape[1]
                assert actual_channels == expected_channels, \
                    f"Channel mismatch: expected {expected_channels}, got {actual_channels}"
            
            # 参数统计
            total_params = sum(p.numel() for p in backbone.parameters())
            trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            print("✓ Test passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n🎉 All backbone tests passed!")
    return True


if __name__ == '__main__':
    test_backbone()