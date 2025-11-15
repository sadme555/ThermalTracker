# models/criterion.py
"""
损失函数实现 - 修复空列表问题版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional
import numpy as np


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器 - 修复空列表问题版本
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
        执行匈牙利匹配 - 修复空列表问题
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 收集所有目标，处理空目标的情况
        tgt_ids_list = []
        tgt_bbox_list = []
        
        for v in targets:
            if len(v["labels"]) > 0:
                tgt_ids_list.append(v["labels"])
                tgt_bbox_list.append(v["boxes"])
        
        # 如果没有有效目标，返回空匹配
        if len(tgt_ids_list) == 0:
            return [(torch.tensor([], dtype=torch.int64, device=outputs["pred_logits"].device), 
                    torch.tensor([], dtype=torch.int64, device=outputs["pred_logits"].device)) 
                    for _ in range(bs)]
        
        # 连接目标
        tgt_ids = torch.cat(tgt_ids_list)
        tgt_bbox = torch.cat(tgt_bbox_list)
        
        # 确保目标边界框是2D的
        if tgt_bbox.dim() == 1:
            tgt_bbox = tgt_bbox.unsqueeze(0)
        
        # 计算分类成本
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        
        # 确保标签在有效范围内
        tgt_ids = torch.clamp(tgt_ids, 0, out_prob.shape[-1] - 1)
        cost_class = -out_prob[:, tgt_ids]
        
        # 计算边界框成本
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # 确保输出边界框是2D的
        if out_bbox.dim() == 1:
            out_bbox = out_bbox.unsqueeze(0)
        
        # 计算距离 - 添加维度检查
        if out_bbox.shape[0] > 0 and tgt_bbox.shape[0] > 0:
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
            # 计算GIoU成本
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
        else:
            # 如果没有有效的边界框，使用零成本
            cost_bbox = torch.zeros((out_bbox.shape[0], tgt_bbox.shape[0]), 
                                  device=out_bbox.device)
            cost_giou = torch.zeros((out_bbox.shape[0], tgt_bbox.shape[0]), 
                                  device=out_bbox.device)
        
        # 修复：检查并处理数值问题
        if torch.isnan(cost_class).any() or torch.isinf(cost_class).any():
            print("Warning: cost_class contains NaN or Inf, replacing with 0")
            cost_class = torch.nan_to_num(cost_class, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(cost_bbox).any() or torch.isinf(cost_bbox).any():
            print("Warning: cost_bbox contains NaN or Inf, replacing with 0")
            cost_bbox = torch.nan_to_num(cost_bbox, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(cost_giou).any() or torch.isinf(cost_giou).any():
            print("Warning: cost_giou contains NaN or Inf, replacing with 0")
            cost_giou = torch.nan_to_num(cost_giou, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        
        # 修复：确保成本矩阵没有无效值
        if torch.isnan(C).any() or torch.isinf(C).any():
            print("Warning: Cost matrix contains NaN or Inf, replacing with large values")
            C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=1e8)
        
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        
        for i, c in enumerate(C.split(sizes, -1)):
            if c[i].numel() > 0 and c[i].shape[1] > 0:
                try:
                    row_ind, col_ind = linear_sum_assignment(c[i])
                    indices.append((torch.as_tensor(row_ind, dtype=torch.int64), 
                                  torch.as_tensor(col_ind, dtype=torch.int64)))
                except Exception as e:
                    print(f"Warning: Hungarian matching failed for sample {i}: {e}")
                    # 如果匹配失败，创建空匹配
                    indices.append((torch.tensor([], dtype=torch.int64), 
                                  torch.tensor([], dtype=torch.int64)))
            else:
                # 如果没有目标，创建空匹配
                indices.append((torch.tensor([], dtype=torch.int64), 
                              torch.tensor([], dtype=torch.int64)))
        
        return indices


class SetCriterion(nn.Module):
    """
    集合预测损失函数 - 修复空列表问题版本
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
        """分类损失 - 修复空列表问题"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        # 收集目标类别
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J) > 0:
                target_classes_o.append(t["labels"][J])
        
        if len(target_classes_o) > 0:
            target_classes_o = torch.cat(target_classes_o)
        else:
            # 如果没有匹配的目标，创建一个空的张量
            target_classes_o = torch.tensor([], dtype=torch.int64, device=src_logits.device)
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        
        if len(idx[0]) > 0 and len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o

        # 修复：检查logits是否包含无效值
        if torch.isnan(src_logits).any() or torch.isinf(src_logits).any():
            print("Warning: src_logits contains NaN or Inf")
            src_logits = torch.nan_to_num(src_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        # 修复：检查损失是否有效
        if torch.isnan(loss_ce) or torch.isinf(loss_ce):
            print("Warning: Classification loss is NaN or Inf, using 0")
            loss_ce = torch.tensor(0.0, device=loss_ce.device)
            
        losses = {'loss_ce': loss_ce}
        
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """边界框损失 - 修复空列表问题"""
        assert 'pred_boxes' in outputs
        
        # 修复：检查预测框是否包含无效值
        if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
            print("Warning: pred_boxes contains NaN or Inf")
            return {'loss_bbox': torch.tensor(0.0, device=outputs['pred_boxes'].device),
                   'loss_giou': torch.tensor(0.0, device=outputs['pred_boxes'].device)}
        
        idx = self._get_src_permutation_idx(indices)
        
        # 检查是否有匹配的源框和目标框
        if len(idx[0]) == 0:
            return {'loss_bbox': torch.tensor(0.0, device=outputs['pred_boxes'].device),
                   'loss_giou': torch.tensor(0.0, device=outputs['pred_boxes'].device)}
        
        src_boxes = outputs['pred_boxes'][idx]
        
        # 收集目标框
        target_boxes_list = []
        for t, (_, i) in zip(targets, indices):
            if len(i) > 0:
                target_boxes_list.append(t['boxes'][i])
        
        if len(target_boxes_list) > 0:
            target_boxes = torch.cat(target_boxes_list)
        else:
            return {'loss_bbox': torch.tensor(0.0, device=src_boxes.device),
                   'loss_giou': torch.tensor(0.0, device=src_boxes.device)}
        
        # 修复：检查边界框是否有效
        if len(src_boxes) == 0 or len(target_boxes) == 0:
            return {'loss_bbox': torch.tensor(0.0, device=src_boxes.device),
                   'loss_giou': torch.tensor(0.0, device=src_boxes.device)}
        
        # 确保维度正确
        if src_boxes.dim() == 1:
            src_boxes = src_boxes.unsqueeze(0)
        if target_boxes.dim() == 1:
            target_boxes = target_boxes.unsqueeze(0)
        
        # 修复：检查边界框坐标是否有效
        if (src_boxes < 0).any() or (src_boxes > 1).any() or \
           (target_boxes < 0).any() or (target_boxes > 1).any():
            print("Warning: Box coordinates out of range [0, 1]")
            return {'loss_bbox': torch.tensor(0.0, device=src_boxes.device),
                   'loss_giou': torch.tensor(0.0, device=src_boxes.device)}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        # 修复：检查L1损失是否有效
        if torch.isnan(loss_bbox).any() or torch.isinf(loss_bbox).any():
            print("Warning: bbox loss contains NaN or Inf")
            loss_bbox = torch.nan_to_num(loss_bbox, nan=0.0, posinf=0.0, neginf=0.0)
            
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        # GIoU损失
        giou_loss = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        
        # 修复：检查GIoU损失是否有效
        if torch.isnan(giou_loss).any() or torch.isinf(giou_loss).any():
            print("Warning: giou loss contains NaN or Inf")
            giou_loss = torch.nan_to_num(giou_loss, nan=0.0, posinf=0.0, neginf=0.0)
            
        losses['loss_giou'] = giou_loss.sum() / num_boxes
        
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
        """获取源排列索引 - 修复空列表问题"""
        # 将匹配对展平为批次索引
        batch_idx_list = []
        src_idx_list = []
        
        for i, (src, _) in enumerate(indices):
            if len(src) > 0:
                batch_idx_list.append(torch.full_like(src, i))
                src_idx_list.append(src)
        
        # 修复：检查列表是否为空
        if len(batch_idx_list) == 0:
            return (torch.tensor([], dtype=torch.int64, device=self.empty_weight.device),
                    torch.tensor([], dtype=torch.int64, device=self.empty_weight.device))
        
        batch_idx = torch.cat(batch_idx_list)
        src_idx = torch.cat(src_idx_list)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """获取目标排列索引 - 修复空列表问题"""
        # 将匹配对展平为目标索引
        batch_idx_list = []
        tgt_idx_list = []
        
        for i, (_, tgt) in enumerate(indices):
            if len(tgt) > 0:
                batch_idx_list.append(torch.full_like(tgt, i))
                tgt_idx_list.append(tgt)
        
        # 修复：检查列表是否为空
        if len(batch_idx_list) == 0:
            return (torch.tensor([], dtype=torch.int64, device=self.empty_weight.device),
                    torch.tensor([], dtype=torch.int64, device=self.empty_weight.device))
        
        batch_idx = torch.cat(batch_idx_list)
        tgt_idx = torch.cat(tgt_idx_list)
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """计算总损失 - 修复空列表问题"""
        # 移除辅助输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 执行匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 计算目标数量用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # 修复：确保num_boxes不为零
        if num_boxes == 0:
            num_boxes = torch.as_tensor([1.0], device=num_boxes.device)
        
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
    # 确保输入是2D的
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """从(x1, y1, x2, y2)转换为(center_x, center_y, width, height)"""
    # 确保输入是2D的
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    计算广义IoU - 修复维度问题版本
    boxes1, boxes2: [N,4], [M,4]
    """
    # 修复：检查输入是否有效
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)
    
    # 确保维度正确
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    
    # 确保boxes1比boxes2小
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 has invalid coordinates"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 has invalid coordinates"
    
    # 交集
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # 并集
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    # 避免除零
    union = torch.clamp(union, min=1e-8)
    iou = inter / union

    # 最小包围框
    lt_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_min = (rb_max - lt_min).clamp(min=0)
    area_min = wh_min[:, :, 0] * wh_min[:, :, 1]

    # 避免除零
    area_min = torch.clamp(area_min, min=1e-8)
    giou = iou - (area_min - union) / area_min
    
    # 修复：确保GIoU值在有效范围内
    giou = torch.clamp(giou, min=-1.0, max=1.0)
    
    return giou