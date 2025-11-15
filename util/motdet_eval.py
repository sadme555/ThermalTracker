import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
import json
import os
from collections import defaultdict
from typing import Dict, List

class InfraredSmallTargetEvaluator:
    """红外小目标检测评估器"""
    
    def __init__(self, iou_thresholds=None, recall_thresholds=None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.recall_thresholds = recall_thresholds or torch.linspace(0, 1, 101).tolist()
        
        # 小目标特定阈值
        self.small_target_threshold = 32 * 32  # 32x32像素以下视为小目标
        
    def evaluate(self, predictions, targets):
        """
        评估预测结果
        
        Args:
            predictions: 预测结果列表
            targets: 真实目标列表
        """
        # 转换为COCO格式进行评估
        coco_results = self._predictions_to_coco_format(predictions)
        coco_targets = self._targets_to_coco_format(targets)
        
        # 计算标准COCO指标
        coco_metrics = self._compute_coco_metrics(coco_results, coco_targets)
        
        # 计算小目标特定指标
        small_target_metrics = self._compute_small_target_metrics(predictions, targets)
        
        # 合并指标
        all_metrics = {**coco_metrics, **small_target_metrics}
        
        return all_metrics
    
    def _predictions_to_coco_format(self, predictions):
        """将预测结果转换为COCO格式"""
        coco_results = []
        for img_id, pred in enumerate(predictions):
            if pred is None:
                continue
                
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for i in range(len(boxes)):
                # 从[x1, y1, x2, y2]转换为[x, y, width, height]
                x1, y1, x2, y2 = boxes[i]
                width = x2 - x1
                height = y2 - y1
                
                coco_results.append({
                    'image_id': img_id,
                    'category_id': int(labels[i]),
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(scores[i])
                })
        
        return coco_results
    
    def _targets_to_coco_format(self, targets):
        """将真实目标转换为COCO格式"""
        coco_targets = []
        for img_id, target in enumerate(targets):
            boxes = target['boxes'].cpu().numpy()
            labels = target['labels'].cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                width = x2 - x1
                height = y2 - y1
                
                coco_targets.append({
                    'image_id': img_id,
                    'category_id': int(labels[i]),
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'area': float(width * height),
                    'iscrowd': 0
                })
        
        return coco_targets
    
    def _compute_coco_metrics(self, results, targets):
        """计算COCO标准指标"""
        # 这里简化实现，实际应该使用pycocotools
        # 在实际使用中，你需要将结果保存为JSON文件并使用COCOeval
        
        metrics = {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0,
        }
        
        # 这里应该是完整的COCO评估实现
        # 为简化，我们返回占位值
        return metrics
    
    def _compute_small_target_metrics(self, predictions, targets):
        """计算小目标特定指标"""
        small_target_recall = self._compute_small_target_recall(predictions, targets)
        small_target_precision = self._compute_small_target_precision(predictions, targets)
        miss_rate = self._compute_miss_rate(predictions, targets)
        
        return {
            'small_target_recall': small_target_recall,
            'small_target_precision': small_target_precision,
            'miss_rate': miss_rate
        }
    
    def _compute_small_target_recall(self, predictions, targets):
        """计算小目标召回率"""
        total_small_targets = 0
        detected_small_targets = 0
        
        for pred, target in zip(predictions, targets):
            if pred is None:
                continue
                
            target_boxes = target['boxes']
            pred_boxes = pred['boxes']
            
            # 筛选小目标
            small_target_mask = self._get_small_target_mask(target_boxes)
            small_targets = target_boxes[small_target_mask]
            total_small_targets += len(small_targets)
            
            if len(pred_boxes) > 0 and len(small_targets) > 0:
                # 计算IoU
                ious = self._box_iou(pred_boxes, small_targets)
                max_ious, _ = torch.max(ious, dim=0)
                
                # 认为IoU > 0.5的匹配成功
                detected_small_targets += (max_ious > 0.5).sum().item()
        
        recall = detected_small_targets / max(total_small_targets, 1)
        return recall
    
    def _compute_small_target_precision(self, predictions, targets):
        """计算小目标精确率"""
        total_small_predictions = 0
        correct_small_predictions = 0
        
        for pred, target in zip(predictions, targets):
            if pred is None:
                continue
                
            pred_boxes = pred['boxes']
            target_boxes = target['boxes']
            
            # 筛选小预测
            small_pred_mask = self._get_small_target_mask(pred_boxes)
            small_predictions = pred_boxes[small_pred_mask]
            total_small_predictions += len(small_predictions)
            
            if len(small_predictions) > 0 and len(target_boxes) > 0:
                # 计算IoU
                ious = self._box_iou(small_predictions, target_boxes)
                max_ious, _ = torch.max(ious, dim=1)
                
                # 认为IoU > 0.5的匹配成功
                correct_small_predictions += (max_ious > 0.5).sum().item()
        
        precision = correct_small_predictions / max(total_small_predictions, 1)
        return precision
    
    def _compute_miss_rate(self, predictions, targets):
        """计算漏检率"""
        total_targets = 0
        missed_targets = 0
        
        for pred, target in zip(predictions, targets):
            target_boxes = target['boxes']
            total_targets += len(target_boxes)
            
            if pred is None or len(pred['boxes']) == 0:
                missed_targets += len(target_boxes)
                continue
                
            pred_boxes = pred['boxes']
            ious = self._box_iou(pred_boxes, target_boxes)
            max_ious, _ = torch.max(ious, dim=0)
            
            # 认为IoU <= 0.5的为漏检
            missed_targets += (max_ious <= 0.5).sum().item()
        
        miss_rate = missed_targets / max(total_targets, 1)
        return miss_rate
    
    def _get_small_target_mask(self, boxes):
        """获取小目标掩码"""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.bool)
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return areas < self.small_target_threshold
    
    def _box_iou(self, boxes1, boxes2):
        """计算两个边界框集合之间的IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        iou = inter / (area1[:, None] + area2 - inter)
        return iou