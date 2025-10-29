# models/transformer.py
"""
Transformer components for Deformable DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is All You Need paper, generalized to work on images.
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


class MultiheadAttention(nn.Module):
    """Multi-head attention module"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        q = self.q_linear(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Attention calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        
        return output


# Simple test function
def test_transformer_components():
    """Test transformer components"""
    print("Testing transformer components...")
    
    # Test PositionEmbeddingSine
    pos_embed = PositionEmbeddingSine(num_pos_feats=128)
    x = torch.randn(2, 256, 32, 32)
    pos = pos_embed(x)
    print(f"Position embedding: {x.shape} -> {pos.shape}")
    
    # Test MultiheadAttention
    attention = MultiheadAttention(d_model=256, nhead=8)
    query = torch.randn(2, 10, 256)  # [batch, seq_len, d_model]
    key = value = torch.randn(2, 20, 256)
    output = attention(query, key, value)
    print(f"Multihead attention: {query.shape} -> {output.shape}")
    
    print("✓ Transformer components test passed!")


if __name__ == '__main__':
    test_transformer_components()