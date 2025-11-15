"""
Deformable DETR模型实现 - 稳定性修复版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, List, Optional, Tuple

from .backbone import build_backbone
from .backbone_config import BackboneConfig


class PositionEmbeddingSine(nn.Module):
    """
    正弦位置编码 - 在deformable_detr.py中直接定义
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MLP(nn.Module):
    """多层感知机 - 用于边界框预测"""
    
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


class DeformableDETR(nn.Module):
    """
    Deformable DETR模型 - 稳定性修复版本
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 骨干网络
        backbone_config = BackboneConfig(
            name=config.backbone,
            pretrained=True,
            train_backbone=True,
            infrared_adaptation=True,
            feature_enhance=True
        )
        self.backbone = build_backbone(backbone_config)
        
        # Transformer
        self.transformer = DeformableTransformer(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            num_encoder_layers=min(config.num_encoder_layers, 4),
            num_decoder_layers=min(config.num_decoder_layers, 4),
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_queries=config.num_queries
        )
        
        # 查询嵌入
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_dim)
        
        # 输入投影层
        if config.backbone == 'resnet50':
            if backbone_config.feature_enhance:
                channel_list = [256]
            else:
                channel_list = [2048]
        else:
            channel_list = [256]
            
        self.input_proj = nn.ModuleList([
            nn.Conv2d(channel, config.hidden_dim, kernel_size=1)
            for channel in channel_list
        ])
        
        # 分类头 - 添加稳定性处理
        self.class_embed = nn.Linear(config.hidden_dim, config.num_classes + 1)
        
        # 边界框回归头 - 添加稳定性处理
        self.bbox_embed = MLP(config.hidden_dim, config.hidden_dim, 4, 3)
        
        # 位置编码
        self.position_encoding = PositionEmbeddingSine(config.hidden_dim // 2, normalize=True)
        
        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数 - 更稳定的初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                # 使用更小的初始化范围
                nn.init.xavier_uniform_(p, gain=0.1)
                
        # 分类头初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)
        
        # 边界框回归头初始化 - 更保守的初始化
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0)
        # 初始化中间层
        for layer in self.bbox_embed.layers[:-1]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)

    def forward(self, samples: torch.Tensor):
        """
        前向传播 - 添加稳定性检查
        """
        # 特征提取
        features = self.backbone(samples)
        
        # 只使用最高层特征以减少内存
        feature_names = list(features.keys())
        last_feature_name = feature_names[-1]
        feature = features[last_feature_name]
        
        # 稳定性检查
        if torch.isnan(feature).any() or torch.isinf(feature).any():
            print("Warning: NaN or Inf in backbone features")
            feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
        
        # 稳定性检查
        if torch.isnan(outputs_class).any() or torch.isinf(outputs_class).any():
            print("Warning: NaN or Inf in outputs_class")
            outputs_class = torch.nan_to_num(outputs_class, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(outputs_coord).any() or torch.isinf(outputs_coord).any():
            print("Warning: NaN or Inf in outputs_coord")
            outputs_coord = torch.nan_to_num(outputs_coord, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 转置维度
        outputs_class = outputs_class[-1].transpose(0, 1)
        outputs_coord = outputs_coord[-1].transpose(0, 1)
        
        out = {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }
        
        return out


class DeformableTransformer(nn.Module):
    """Deformable Transformer - 稳定性修复版本"""
    
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1,
                 num_queries=100):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        
        # 编码器 - 减少层数以提高稳定性
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, min(num_encoder_layers, 2))
        
        # 解码器 - 减少层数以提高稳定性
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, min(num_decoder_layers, 2), num_queries)

    def forward(self, srcs, masks, query_embed, pos_embeds):
        """
        前向传播 - 添加稳定性检查
        """
        # 使用最高层特征
        src = srcs[0]
        mask = masks[0]
        pos = pos_embeds[0]
        
        # 稳定性检查
        if torch.isnan(src).any() or torch.isinf(src).any():
            print("Warning: NaN or Inf in transformer input src")
            src = torch.nan_to_num(src, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            print("Warning: NaN or Inf in transformer input pos")
            pos = torch.nan_to_num(pos, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 展平空间维度
        batch_size, channels, height, width = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [seq_len, batch_size, channels]
        mask = mask.flatten(1)  # [batch_size, seq_len]
        pos = pos.flatten(2).permute(2, 0, 1)  # [seq_len, batch_size, channels]
        
        # 准备查询
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, batch_size, d_model]
        
        # 通过编码器
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        
        # 通过解码器
        hs = self.decoder(query_embed, memory, memory_key_padding_mask=mask,
                         pos=pos, query_pos=query_embed)
        
        return hs


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层 - 稳定性修复版本"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src, src_key_padding_mask=None, pos=None):
        # 添加位置编码
        q = k = src if pos is None else src + pos
        
        # 自注意力
        src2, attn_weights = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        
        # 稳定性检查
        if torch.isnan(src2).any() or torch.isinf(src2).any():
            print("Warning: NaN or Inf in encoder attention output")
            src2 = torch.nan_to_num(src2, nan=0.0, posinf=1.0, neginf=-1.0)
            
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # 稳定性检查
        if torch.isnan(src2).any() or torch.isinf(src2).any():
            print("Warning: NaN or Inf in encoder FFN output")
            src2 = torch.nan_to_num(src2, nan=0.0, posinf=1.0, neginf=-1.0)
            
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
    """Transformer解码器层 - 稳定性修复版本"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
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
        
        tgt2, self_attn_weights = self.self_attn(q, k, value=tgt, key_padding_mask=None)
        
        # 稳定性检查
        if torch.isnan(tgt2).any() or torch.isinf(tgt2).any():
            print("Warning: NaN or Inf in decoder self-attention output")
            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=1.0, neginf=-1.0)
            
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        
        tgt2, cross_attn_weights = self.multihead_attn(
            query=q,
            key=k,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        
        # 稳定性检查
        if torch.isnan(tgt2).any() or torch.isinf(tgt2).any():
            print("Warning: NaN or Inf in decoder cross-attention output")
            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=1.0, neginf=-1.0)
            
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
        # 稳定性检查
        if torch.isnan(tgt2).any() or torch.isinf(tgt2).any():
            print("Warning: NaN or Inf in decoder FFN output")
            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=1.0, neginf=-1.0)
            
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
        
        # 返回所有层的输出
        return torch.stack(intermediate)