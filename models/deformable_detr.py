# models/deformable_detr.py
"""
Deformable DETR模型实现 - 内存优化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Optional, Tuple

from .backbone import build_backbone
from .backbone_config import BackboneConfig
from .transformer import PositionEmbeddingSine, MultiheadAttention


class DeformableDETR(nn.Module):
    """
    Deformable DETR模型 - 内存优化版本
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 骨干网络 - 使用较小的配置以减少内存
        backbone_config = BackboneConfig(
            name=config.backbone,
            pretrained=True,
            train_backbone=True,
            infrared_adaptation=True,
            feature_enhance=True
        )
        self.backbone = build_backbone(backbone_config)
        
        # Transformer - 使用更少的层数以减少内存
        self.transformer = DeformableTransformer(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            num_encoder_layers=min(config.num_encoder_layers, 4),  # 限制最大层数
            num_decoder_layers=min(config.num_decoder_layers, 4),
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_queries=config.num_queries
        )
        
        # 查询嵌入
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)
        
        # 输入投影层 - 使用更少的特征层级
        if config.backbone == 'resnet50':
            if backbone_config.feature_enhance:
                channel_list = [256]  # 只使用一个特征层级以减少内存
            else:
                channel_list = [2048]  # 只使用最高层特征
        else:
            channel_list = [256]
            
        self.input_proj = nn.ModuleList([
            nn.Conv2d(channel, config.hidden_dim, kernel_size=1)
            for channel in channel_list
        ])
        
        # 分类头
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        
        # 边界框回归头
        self.bbox_embed = MLP(config.hidden_dim, config.hidden_dim, 4, 3)
        
        # 位置编码
        self.position_encoding = PositionEmbeddingSine(config.hidden_dim // 2, normalize=True)
        
        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)
        
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0)

    def forward(self, samples: torch.Tensor):
        """
        前向传播 - 内存优化版本
        """
        # 特征提取
        features = self.backbone(samples)
        
        # 只使用最高层特征以减少内存
        feature_names = list(features.keys())
        last_feature_name = feature_names[-1]  # 使用最后一个特征层级
        feature = features[last_feature_name]
        
        # 准备Transformer输入
        src = self.input_proj[0](feature)
        mask = torch.zeros((src.shape[0], src.shape[2], src.shape[3]), 
                         dtype=torch.bool, device=src.device)
        pos = self.position_encoding(src, mask)
        
        # 通过Transformer
        hs = self.transformer([src], [mask], self.query_embed.weight, [pos])
        
        # 输出预测
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        # 转置维度: [num_layers, num_queries, batch, ...] -> [batch, num_queries, ...]
        outputs_class = outputs_class[-1].transpose(0, 1)
        outputs_coord = outputs_coord[-1].transpose(0, 1)
        
        out = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
        
        return out


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
        return x


class DeformableTransformer(nn.Module):
    """Deformable Transformer - 内存优化版本"""
    
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1,
                 num_queries=100):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        
        # 编码器 - 使用更少的层数
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 解码器 - 使用更少的层数
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, num_queries)

    def forward(self, srcs, masks, query_embed, pos_embeds):
        """
        前向传播 - 内存优化版本
        """
        # 使用最高层特征
        src = srcs[0]
        mask = masks[0]
        pos = pos_embeds[0]
        
        # 展平空间维度 [batch, channels, H, W] -> [H*W, batch, channels]
        batch_size, channels, height, width = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, batch, channels]
        mask = mask.flatten(1)  # [batch, H*W]
        pos = pos.flatten(2).permute(2, 0, 1)  # [H*W, batch, channels]
        
        # 准备查询 [num_queries, batch, channels]
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        
        # 通过编码器
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        
        # 通过解码器
        hs = self.decoder(query_embed, memory, memory_key_padding_mask=mask,
                         pos=pos, query_pos=query_embed)
        
        return hs  # [num_layers, num_queries, batch, channels]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src, src_key_padding_mask=None, pos=None):
        # 自注意力
        q = k = src if pos is None else src + pos
        
        # 转置以适配我们的MultiheadAttention
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        src_t = src.transpose(0, 1)
        
        src2 = self.self_attn(q, k, value=src_t, attn_mask=None)
        src2 = src2.transpose(0, 1)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        # 自注意力
        q = k = tgt if query_pos is None else tgt + query_pos
        
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        tgt_t = tgt.transpose(0, 1)
        
        tgt2 = self.self_attn(q_t, k_t, value=tgt_t)
        tgt2 = tgt2.transpose(0, 1)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        q = tgt if query_pos is None else tgt + query_pos
        
        q_t = q.transpose(0, 1)
        memory_t = memory.transpose(0, 1)
        pos_t = pos.transpose(0, 1) if pos is not None else None
        
        tgt2 = self.multihead_attn(
            query=q_t,
            key=memory_t if pos is None else memory_t + pos_t,
            value=memory_t
        )
        tgt2 = tgt2.transpose(0, 1)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, decoder_layer, num_layers, num_queries):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_queries = num_queries

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt
        
        # 确保输出有正确的查询数量
        assert output.shape[0] == self.num_queries, f"Query count mismatch: expected {self.num_queries}, got {output.shape[0]}"
        
        # 存储所有层的输出
        intermediate = []
        
        for layer in self.layers:
            output = layer(
                output, memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, 
                query_pos=query_pos
            )
            intermediate.append(output)
        
        # 返回所有层的输出 [num_layers, num_queries, batch, d_model]
        return torch.stack(intermediate)


# 测试函数 - 使用内存优化配置
def test_deformable_detr():
    """测试Deformable DETR模型 - 内存优化版本"""
    print("=" * 60)
    print("Testing Deformable DETR Model (Memory Optimized)")
    print("=" * 60)
    
    # 创建内存优化配置
    class Config:
        backbone = 'resnet50'
        hidden_dim = 256
        num_queries = 50  # 减少查询数量
        num_classes = 7
        nheads = 8
        num_encoder_layers = 4  # 减少层数
        num_decoder_layers = 4
        dim_feedforward = 1024
        dropout = 0.1
    
    config = Config()
    
    try:
        # 创建模型
        print("▶ Building model...")
        model = DeformableDETR(config)
        
        # 测试输入 - 使用较小尺寸
        x = torch.randn(2, 3, 384, 480)  # 较小尺寸
        print(f"Input shape: {x.shape}")
        
        # 前向传播
        print("▶ Forward pass...")
        with torch.no_grad():
            outputs = model(x)
        
        print("✓ Model test passed!")
        print(f"Output keys: {list(outputs.keys())}")
        print(f"pred_logits shape: {outputs['pred_logits'].shape}")
        print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
        
        # 验证形状
        batch_size = x.shape[0]
        assert outputs['pred_logits'].shape == (batch_size, config.num_queries, config.num_classes + 1), \
            f"pred_logits shape mismatch: {outputs['pred_logits'].shape}"
        assert outputs['pred_boxes'].shape == (batch_size, config.num_queries, 4), \
            f"pred_boxes shape mismatch: {outputs['pred_boxes'].shape}"
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model, outputs
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    model, outputs = test_deformable_detr()