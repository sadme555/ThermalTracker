"""
评估模块 - 实现详细的检测评估
"""

import torch
import numpy as np
from util.motdet_eval import InfraredSmallTargetEvaluator

def evaluate_detection(model, data_loader, device, config):
    """
    详细的检测评估
    """
    model.eval()
    evaluator = InfraredSmallTargetEvaluator()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            outputs = model(images)
            
            # 转换预测格式
            pred_boxes = outputs['pred_boxes']
            pred_scores = outputs['pred_logits'].softmax(-1)[:, :, :-1].max(-1)[0]
            pred_labels = outputs['pred_logits'].softmax(-1)[:, :, :-1].max(-1)[1]
            
            # 应用置信度阈值
            keep = pred_scores > config.evaluation.confidence_threshold
            batch_predictions = []
            
            for i in range(len(images)):
                if keep[i].sum() > 0:
                    batch_pred = {
                        'boxes': pred_boxes[i][keep[i]].cpu(),
                        'scores': pred_scores[i][keep[i]].cpu(),
                        'labels': pred_labels[i][keep[i]].cpu()
                    }
                else:
                    batch_pred = None
                batch_predictions.append(batch_pred)
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(targets)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx}/{len(data_loader)} batches")
    
    # 计算评估指标
    metrics = evaluator.evaluate(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def evaluate_model_detailed(model, data_loader, device, config):
    """
    详细的模型评估，包括各种指标
    """
    print("Starting detailed evaluation...")
    
    metrics, predictions, targets = evaluate_detection(model, data_loader, device, config)
    
    print("\n" + "="*50)
    print("Detailed Evaluation Results")
    print("="*50)
    
    # 分类指标
    print("\nClassification Metrics:")
    for key in ['loss_ce', 'loss_bbox', 'loss_giou']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    # 检测指标
    print("\nDetection Metrics:")
    for key in ['mAP', 'mAP_50', 'mAP_75']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    # 小目标特定指标
    print("\nSmall Target Metrics:")
    for key in ['small_target_recall', 'small_target_precision', 'miss_rate']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    return metrics