# models/criterion.py
"""
Deformable DETR损失函数 - 匈牙利匹配损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional
import numpy as np


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    将预测与真实标注进行一对一匹配
    """
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "所有损失权重不能同时为0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        执行匈牙利匹配
        
        Args:
            outputs: 模型输出，包含pred_logits和pred_boxes
            targets: 真实标注列表
            
        Returns:
            indices: 匹配索引列表
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 将输出拆分为每个样本
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # 为每个样本计算匹配成本
        indices = []
        for i in range(bs):
            target = targets[i]
            if len(target["boxes"]) == 0:
                # 如果没有真实目标，将所有预测匹配到"无目标"
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue
                
            # 获取当前样本的目标
            tgt_ids = target["labels"]
            tgt_bbox = target["boxes"]
            
            # 分类成本：负对数似然
            cost_class = -out_prob[i * num_queries:(i + 1) * num_queries, tgt_ids]
            
            # 边界框L1成本
            cost_bbox = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)
            
            # GIoU成本
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[i * num_queries:(i + 1) * num_queries]),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            
            # 总成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.reshape(num_queries, -1).cpu()
            
            # 匈牙利算法
            indices_i = linear_sum_assignment(C)
            indices.append((torch.as_tensor(indices_i[0], dtype=torch.int64), 
                           torch.as_tensor(indices_i[1], dtype=torch.int64)))
        
        return indices


class SetCriterion(nn.Module):
    """
    集合预测损失函数
    """
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 空类别权重
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """分类损失"""
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """边界框损失（L1 + GIoU）"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        
        # GIoU损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # 获取匹配的源索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 获取匹配的目标索引
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """
        计算总损失
        
        Args:
            outputs: 模型输出
            targets: 真实标注列表
            
        Returns:
            loss_dict: 损失字典
        """
        # 匹配预测和真实标注
        indices = self.matcher(outputs, targets)
        
        # 计算目标框数量（用于归一化）
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 计算各项损失
        loss_dict = {}
        for loss in self.losses:
            loss_dict.update(getattr(self, f'loss_{loss}')(outputs, targets, indices, num_boxes))
        
        return loss_dict


# 工具函数
def box_cxcywh_to_xyxy(x):
    """将边界框从(center_x, center_y, width, height)格式转换为(x1, y1, x2, y2)格式"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """将边界框从(x1, y1, x2, y2)格式转换为(center_x, center_y, width, height)格式"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    计算广义IoU
    boxes1, boxes2: [N,4], [M,4]
    """
    # 交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # 并集
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    
    # 最小包围框
    lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_min = (rb_max - lt_min).clamp(min=0)
    area_min = wh_min[:, :, 0] * wh_min[:, :, 1]
    
    giou = iou - (area_min - union) / area_min
    
    return giou


# 测试函数
def test_criterion():
    """测试损失函数"""
    print("Testing criterion...")
    
    # 创建匹配器
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    
    # 创建损失函数
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(
        num_classes=7,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['labels', 'boxes']
    )
    
    # 模拟输出
    outputs = {
        'pred_logits': torch.randn(2, 100, 8),  # [batch, queries, classes+1]
        'pred_boxes': torch.rand(2, 100, 4)     # [batch, queries, 4]
    }
    
    # 模拟目标
    targets = [
        {
            'labels': torch.tensor([0, 1, 2]),  # 3个目标
            'boxes': torch.rand(3, 4)           # 归一化坐标
        },
        {
            'labels': torch.tensor([3, 4]),     # 2个目标
            'boxes': torch.rand(2, 4)
        }
    ]
    
    # 计算损失
    loss_dict = criterion(outputs, targets)
    
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("✓ Criterion test passed!")
    return criterion, loss_dict


if __name__ == '__main__':
    test_criterion()