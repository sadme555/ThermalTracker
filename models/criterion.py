# models/criterion.py
"""
损失函数实现 - 完整的匈牙利匹配和集合预测损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional
import numpy as np


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器 - 将预测与真实目标进行最优匹配
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
            outputs: 模型输出字典
            targets: 真实目标列表
            
        Returns:
            matches: 匹配结果列表
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 将目标展平为单个批次
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 计算分类成本
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        cost_class = -out_prob[:, tgt_ids]
        
        # 计算边界框成本
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 计算GIoU成本
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """
    集合预测损失函数
    """
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1, losses=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        if losses is None:
            losses = ['labels', 'boxes', 'cardinality']
        self.losses = losses
        
        # 空类权重
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """分类损失"""
        assert 'pred_logits' in outputs
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
        """边界框损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """基数损失 - 鼓励预测正确数量的目标"""
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def _get_src_permutation_idx(self, indices):
        # 将匹配对展平为批次索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 将匹配对展平为目标索引
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """计算总损失"""
        # 移除辅助输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 执行匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 计算目标数量用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 收集所有损失
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets, indices, num_boxes))

        # 处理辅助损失
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = getattr(self, f'loss_{loss}')(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


# 工具函数
def box_cxcywh_to_xyxy(x):
    """从(center_x, center_y, width, height)转换为(x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """从(x1, y1, x2, y2)转换为(center_x, center_y, width, height)"""
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
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union

    # 最小包围框
    lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_min = (rb_max - lt_min).clamp(min=0)
    area_min = wh_min[:, :, 0] * wh_min[:, :, 1]

    return iou - (area_min - union) / area_min


# 测试函数
def test_criterion():
    """测试损失函数"""
    print("Testing criterion components...")
    
    # 测试边界框转换
    boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]])  # cxcywh
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes_back = box_xyxy_to_cxcywh(boxes_xyxy)
    assert torch.allclose(boxes, boxes_back), "Box conversion failed"
    print("✓ Box conversion test passed")
    
    # 测试GIoU
    box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    box2 = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
    giou = generalized_box_iou(box1, box2)
    print(f"✓ GIoU test passed: {giou.item():.3f}")
    
    # 测试匹配器
    matcher = HungarianMatcher()
    print("✓ Matcher creation test passed")
    
    # 测试损失函数
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(
        num_classes=7,
        matcher=matcher,
        weight_dict=weight_dict
    )
    print("✓ Criterion creation test passed")
    
    print("🎉 All criterion tests passed!")


if __name__ == '__main__':
    test_criterion()