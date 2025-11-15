# engine/train.py
"""
训练引擎 - 修复版本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
from typing import Dict, List
from util.misc import MetricLogger, SmoothedValue
from util.misc import reduce_dict


def train_one_epoch(model: nn.Module, criterion: nn.Module, 
                   data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   device: torch.device, epoch: int, max_norm: float = 0.1,
                   print_freq: int = 10):
    """
    训练一个epoch - 修复版本
    """
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            # 安全检查1: 检查输入数据
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"⚠️  Batch {batch_idx}: images contain NaN/Inf, skipping")
                continue
            
            # 移动到设备
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 安全检查2: 验证目标数据
            valid_targets = []
            for i, target in enumerate(targets):
                if 'boxes' in target and len(target['boxes']) > 0:
                    boxes = target['boxes']
                    # 确保边界框有效
                    if (boxes < 0).any() or (boxes > 1).any():
                        # 修复边界框
                        target['boxes'] = boxes.clamp(0, 1)
                    valid_targets.append(target)
                else:
                    # 没有目标的情况
                    valid_targets.append({
                        'labels': torch.tensor([], device=device, dtype=torch.int64),
                        'boxes': torch.tensor([], device=device, dtype=torch.float32)
                    })
            
            # 前向传播
            outputs = model(images)
            
            # 安全检查3: 检查模型输出
            if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                print(f"⚠️  Batch {batch_idx}: model outputs contain NaN/Inf, skipping")
                optimizer.zero_grad()
                continue
                
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(f"⚠️  Batch {batch_idx}: model boxes contain NaN/Inf, skipping")
                optimizer.zero_grad()
                continue
            
            # 计算损失
            loss_dict = criterion(outputs, valid_targets)
            
            # 安全检查4: 检查损失值
            valid_loss = True
            for key, value in loss_dict.items():
                if torch.isnan(value) or torch.isinf(value):
                    print(f"⚠️  Batch {batch_idx}: {key} loss is NaN/Inf")
                    valid_loss = False
                    break
            
            if not valid_loss:
                print(f"⚠️  Batch {batch_idx}: invalid loss, skipping")
                optimizer.zero_grad()
                continue
            
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # 减少所有进程的损失
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss_dict_reduced[k] * weight_dict[k] for k in loss_dict_reduced.keys() if k in weight_dict)
            
            loss_value = losses_reduced.item()
            
            # 检查损失是否有效
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                exit(1)
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            
            # 梯度检查
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if torch.isnan(torch.tensor(total_norm)).any() or torch.isinf(torch.tensor(total_norm)).any():
                print(f"⚠️  Batch {batch_idx}: gradient contains NaN/Inf, skipping update")
                optimizer.zero_grad()
                continue
            
            # 梯度裁剪
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            # 记录指标 - 修复：确保所有值都是标量
            metric_logger.update(loss=loss_value)
            for k, v in loss_dict_reduced.items():
                metric_logger.update(**{k: v.item() if torch.is_tensor(v) else v})
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
        except Exception as e:
            print(f"❌ Batch {batch_idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            optimizer.zero_grad()
            continue
    
    # 在所有进程上汇总指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model: nn.Module, criterion: nn.Module, data_loader: DataLoader, 
            device: torch.device, print_freq: int = 10):
    """
    评估模型
    """
    model.eval()
    criterion.eval()
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            
            # 减少损失
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss_dict_reduced[k] * weight_dict[k] for k in loss_dict_reduced.keys() if k in weight_dict)
            loss_value = losses_reduced.item()
            
            metric_logger.update(loss=loss_value)
            for k, v in loss_dict_reduced.items():
                metric_logger.update(**{k: v.item() if torch.is_tensor(v) else v})
    
    # 在所有进程上汇总指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class EarlyStopping:
    """早停机制 - 修复版本"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, val_loss):
        """检查是否应该早停"""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.status = f"Initial loss: {val_loss:.4f}"
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
            return False
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                return True
        return False
    
    def save_checkpoint(self, model):
        """保存最佳模型状态"""
        if self.restore_best_weights:
            self.best_model_state = model.state_dict().copy()
    
    def restore_best_weights(self, model):
        """恢复最佳模型权重"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("Restored best model weights")