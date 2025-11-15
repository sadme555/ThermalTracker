"""
Transformer components for Deformable DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    
    # Test MultiheadAttention
    attention = MultiheadAttention(d_model=256, nhead=8)
    query = torch.randn(2, 10, 256)  # [batch, seq_len, d_model]
    key = value = torch.randn(2, 20, 256)
    output = attention(query, key, value)
    print(f"Multihead attention: {query.shape} -> {output.shape}")
    
    print("✓ Transformer components test passed!")


if __name__ == '__main__':
    test_transformer_components()