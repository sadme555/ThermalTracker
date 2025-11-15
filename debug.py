#!/usr/bin/env python3
"""
详细调试数值问题
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import build_infrared_dataset, collate_fn
from models import DeformableDETR, SetCriterion, HungarianMatcher
from torch.utils.data import DataLoader


def debug_detailed():
    """详细调试"""
    print("Detailed debugging...")
    
    try:
        # 创建简单的配置对象
        class SimpleConfig:
            class DataConfig:
                dataset_root = "datasets/RGBT-Tiny"
                num_classes = 7
                image_size = (512, 640)
            
            class ModelConfig:
                backbone = "resnet50"
                hidden_dim = 256
                num_queries = 10  # 进一步减少查询数量
                num_classes = 7
                nheads = 8
                num_encoder_layers = 2
                num_decoder_layers = 2
                dim_feedforward = 512
                dropout = 0.1
                num_feature_levels = 3
            
            class LossConfig:
                cost_class = 1.0
                cost_bbox = 5.0
                cost_giou = 2.0
                weight_class = 1.0
                weight_bbox = 5.0
                weight_giou = 2.0
                eos_coef = 0.1
            
            class TrainingConfig:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                batch_size = 1  # 使用批次大小为1
                num_workers = 0
            
            data = DataConfig()
            model = ModelConfig()
            loss = LossConfig()
            training = TrainingConfig()
        
        config = SimpleConfig()
        
        # 构建最小数据集
        print("Building minimal dataset...")
        dataset = build_infrared_dataset(config, is_train=True)
        
        # 只使用1个样本
        from torch.utils.data import Subset
        debug_dataset = Subset(dataset, [0])
        
        dataloader = DataLoader(
            debug_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=collate_fn
        )
        
        # 构建模型
        print("Building model...")
        model = DeformableDETR(config.model)
        model = model.to(config.training.device)
        
        # 构建损失函数
        print("Building criterion...")
        matcher = HungarianMatcher(
            cost_class=config.loss.cost_class,
            cost_bbox=config.loss.cost_bbox,
            cost_giou=config.loss.cost_giou
        )
        
        weight_dict = {
            'loss_ce': config.loss.weight_class,
            'loss_bbox': config.loss.weight_bbox,
            'loss_giou': config.loss.weight_giou
        }
        
        criterion = SetCriterion(
            num_classes=config.data.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=config.loss.eos_coef
        )
        criterion = criterion.to(config.training.device)
        
        # 详细调试
        print("Detailed debugging...")
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                print(f"\n=== Batch {batch_idx} ===")
                
                # 检查输入数据
                print("Input data check:")
                print(f"  Images shape: {images.shape}")
                print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
                
                # 检查目标数据
                for i, target in enumerate(targets):
                    print(f"  Target {i}:")
                    print(f"    boxes: {target['boxes'].shape}")
                    print(f"    boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
                    print(f"    labels: {target['labels'].shape}")
                    print(f"    labels: {target['labels']}")
                
                # 前向传播
                images = images.to(config.training.device)
                targets_device = [{k: v.to(config.training.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in t.items()} for t in targets]
                
                outputs = model(images)
                
                # 检查模型输出
                print("\nModel outputs:")
                for key, value in outputs.items():
                    if key != 'aux_outputs':
                        print(f"  {key}: {value.shape}")
                        print(f"    range: [{value.min():.3f}, {value.max():.3f}]")
                        
                        if torch.isnan(value).any():
                            nan_count = torch.isnan(value).sum().item()
                            print(f"    ❌ Contains {nan_count} NaN values")
                        
                        if torch.isinf(value).any():
                            inf_count = torch.isinf(value).sum().item()
                            print(f"    ❌ Contains {inf_count} Inf values")
                
                # 手动执行匹配过程来调试
                print("\nManual matching debug:")
                try:
                    # 执行匹配
                    indices = matcher(outputs, targets_device)
                    print("  ✅ Matching successful")
                    print(f"  Indices: {indices}")
                except Exception as e:
                    print(f"  ❌ Matching failed: {e}")
                    
                    # 详细调试匹配过程
                    print("\n  Detailed matching debug:")
                    bs, num_queries = outputs["pred_logits"].shape[:2]
                    
                    # 将目标展平为单个批次
                    tgt_ids = torch.cat([v["labels"] for v in targets_device])
                    tgt_bbox = torch.cat([v["boxes"] for v in targets_device])
                    
                    print(f"  tgt_ids: {tgt_ids.shape}, {tgt_ids}")
                    print(f"  tgt_bbox: {tgt_bbox.shape}, range: [{tgt_bbox.min():.3f}, {tgt_bbox.max():.3f}]")
                    
                    # 计算分类成本
                    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
                    cost_class = -out_prob[:, tgt_ids]
                    print(f"  cost_class: {cost_class.shape}, range: [{cost_class.min():.3f}, {cost_class.max():.3f}]")
                    
                    # 计算边界框成本
                    out_bbox = outputs["pred_boxes"].flatten(0, 1)
                    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                    print(f"  cost_bbox: {cost_bbox.shape}, range: [{cost_bbox.min():.3f}, {cost_bbox.max():.3f}]")
                    
                    # 计算GIoU成本
                    from models.criterion import generalized_box_iou, box_cxcywh_to_xyxy
                    cost_giou = -generalized_box_iou(
                        box_cxcywh_to_xyxy(out_bbox),
                        box_cxcywh_to_xyxy(tgt_bbox)
                    )
                    print(f"  cost_giou: {cost_giou.shape}, range: [{cost_giou.min():.3f}, {cost_giou.max():.3f}]")
                    
                    # 最终成本矩阵
                    C = (matcher.cost_bbox * cost_bbox + 
                         matcher.cost_class * cost_class + 
                         matcher.cost_giou * cost_giou)
                    print(f"  C: {C.shape}, range: [{C.min():.3f}, {C.max():.3f}]")
                
                # 尝试计算损失
                print("\nLoss calculation:")
                try:
                    loss_dict = criterion(outputs, targets_device)
                    print("  ✅ Loss calculation successful")
                    for key, value in loss_dict.items():
                        print(f"    {key}: {value.item():.6f}")
                except Exception as e:
                    print(f"  ❌ Loss calculation failed: {e}")
                
                break  # 只测试一个批次
        
        print("\n🎉 Detailed debug completed!")
        return True
        
    except Exception as e:
        print(f"❌ Detailed debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    debug_detailed()