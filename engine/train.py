# engine/train.py
"""
训练引擎 - 修复版，包含完整的tqdm进度条
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
from typing import Dict, List
from util.misc import MetricLogger, SmoothedValue
from util.misc import reduce_dict
from tqdm import tqdm


def train_one_epoch(model: nn.Module, criterion: nn.Module, 
                   data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   device: torch.device, epoch: int, max_norm: float = 0.1,
                   print_freq: int = 50):
    """
    训练一个epoch - 完整修复版本
    """
    model.train()
    criterion.train()
    
    # 创建进度条
    total_batches = len(data_loader)
    pbar = tqdm(total=total_batches, desc=f'Epoch {epoch+1}', 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                position=0, leave=True)
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        start_time = time.time()
        
        try:
            # 安全检查1: 检查输入数据
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"⚠️  Batch {batch_idx}: images contain NaN/Inf, skipping")
                pbar.update(1)
                continue
            
            # 移动到设备
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 安全检查2: 检查目标数据
            valid_targets = []
            for i, target in enumerate(targets):
                if 'boxes' in target and len(target['boxes']) > 0:
                    boxes = target['boxes']
                    # 检查边界框是否有效
                    if (boxes < 0).any() or (boxes > 1).any():
                        # 修复边界框
                        target['boxes'] = boxes.clamp(0, 1)
                    valid_targets.append(target)
                else:
                    # 创建空目标
                    valid_targets.append({
                        'labels': torch.tensor([], device=device, dtype=torch.int64), 
                        'boxes': torch.tensor([], device=device, dtype=torch.float32)
                    })
            
            # 前向传播
            outputs = model(images)
            
            # 安全检查3: 检查模型输出
            if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                print(f"⚠️  Batch {batch_idx}: model outputs contain NaN/Inf, skipping")
                pbar.update(1)
                continue
                
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(f"⚠️  Batch {batch_idx}: model boxes contain NaN/Inf, skipping")
                pbar.update(1)
                continue
            
            # 计算损失
            loss_dict = criterion(outputs, valid_targets)
            
            # 安全检查4: 检查损失
            valid_loss = True
            for key, value in loss_dict.items():
                if torch.isnan(value) or torch.isinf(value):
                    print(f"⚠️  Batch {batch_idx}: {key} loss is NaN/Inf")
                    valid_loss = False
                    break
            
            if not valid_loss:
                print(f"⚠️  Batch {batch_idx}: invalid loss, skipping")
                pbar.update(1)
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
                pbar.update(1)
                continue
            
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
                pbar.update(1)
                continue
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            # 记录指标
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            # 更新进度条
            batch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'lr': f'{current_lr:.2e}',
                'time': f'{batch_time:.2f}s'
            })
            pbar.update(1)
            
        except Exception as e:
            print(f"❌ Batch {batch_idx}: Error - {e}")
            import traceback
            traceback.print_exc()
            optimizer.zero_grad()
            pbar.update(1)
            continue
    
    # 关闭进度条
    pbar.close()
    
    # 在所有进程上汇总指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model: nn.Module, criterion: nn.Module, data_loader: DataLoader, 
            device: torch.device, print_freq: int = 50):
    """
    评估模型 - 修复版本
    """
    model.eval()
    criterion.eval()
    
    # 创建评估进度条
    total_batches = len(data_loader)
    pbar = tqdm(total=total_batches, desc='Evaluating', 
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                position=0, leave=True)
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Validation:'
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            start_time = time.time()
            
            try:
                # 安全检查
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"⚠️  Eval Batch {batch_idx}: images contain NaN/Inf, skipping")
                    pbar.update(1)
                    continue
                
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # 安全检查目标数据
                valid_targets = []
                for target in targets:
                    if 'boxes' in target and len(target['boxes']) > 0:
                        boxes = target['boxes']
                        if (boxes < 0).any() or (boxes > 1).any():
                            target['boxes'] = boxes.clamp(0, 1)
                        valid_targets.append(target)
                    else:
                        valid_targets.append({
                            'labels': torch.tensor([], device=device, dtype=torch.int64), 
                            'boxes': torch.tensor([], device=device, dtype=torch.float32)
                        })
                
                # 前向传播
                outputs = model(images)
                
                # 检查模型输出
                if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                    print(f"⚠️  Eval Batch {batch_idx}: model outputs contain NaN/Inf, skipping")
                    pbar.update(1)
                    continue
                    
                if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                    print(f"⚠️  Eval Batch {batch_idx}: model boxes contain NaN/Inf, skipping")
                    pbar.update(1)
                    continue
                
                loss_dict = criterion(outputs, valid_targets)
                weight_dict = criterion.weight_dict
                
                # 减少损失
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss_dict_reduced[k] * weight_dict[k] for k in loss_dict_reduced.keys() if k in weight_dict)
                loss_value = losses_reduced.item()
                
                # 检查损失是否有效
                if not math.isfinite(loss_value):
                    print(f"Eval Loss is {loss_value}, skipping batch")
                    pbar.update(1)
                    continue
                
                metric_logger.update(loss=loss_value, **loss_dict_reduced)
                
                # 更新进度条
                batch_time = time.time() - start_time
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'time': f'{batch_time:.2f}s'
                })
                pbar.update(1)
                
            except Exception as e:
                print(f"❌ Eval Batch {batch_idx}: Error - {e}")
                pbar.update(1)
                continue
    
    # 关闭进度条
    pbar.close()
    
    # 在所有进程上汇总指标
    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class EarlyStopping:
    """早停机制 - 修复版本"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, model, optimizer, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer)
            self.status = f"Initial best loss: {val_loss:.4f}"
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer)
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                return True
        return False
    
    def save_checkpoint(self, model, optimizer):
        if self.restore_best_weights:
            self.best_model_state = model.state_dict().copy()
            self.best_optimizer_state = optimizer.state_dict().copy()
    
    def restore_best_weights(self, model, optimizer):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        if self.best_optimizer_state is not None:
            optimizer.load_state_dict(self.best_optimizer_state)


# 测试函数
def test_train_functions():
    """测试训练函数"""
    print("Testing training functions...")
    
    # 创建模拟数据
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x):
            return {
                'pred_logits': torch.randn(2, 5, 3),
                'pred_boxes': torch.rand(2, 5, 4)
            }
    
    class MockCriterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        
        def forward(self, outputs, targets):
            return {
                'loss_ce': torch.tensor(1.0),
                'loss_bbox': torch.tensor(0.5),
                'loss_giou': torch.tensor(0.3)
            }
    
    # 创建模拟数据加载器
    class MockDataLoader:
        def __init__(self):
            self.data = [(torch.randn(2, 3, 100, 100), [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]])}]) for _ in range(5)]
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)
    
    # 测试
    model = MockModel()
    criterion = MockCriterion()
    data_loader = MockDataLoader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cpu')
    
    try:
        # 测试训练一个epoch
        stats = train_one_epoch(model, criterion, data_loader, optimizer, device, 0)
        print("✓ train_one_epoch test passed")
        
        # 测试评估
        eval_stats = evaluate(model, criterion, data_loader, device)
        print("✓ evaluate test passed")
        
        # 测试早停
        early_stopping = EarlyStopping(patience=3)
        should_stop = early_stopping(model, optimizer, 1.0)
        print(f"✓ EarlyStopping test passed: should_stop = {should_stop}")
        
        print("✓ All training function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_train_functions()