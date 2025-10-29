# engine/trainer.py
"""
训练器实现
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import json


class Trainer:
    def __init__(self, model, criterion, optimizer, lr_scheduler, device, output_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, data_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_loss_dict = {}
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(torch.stack(images))
            
            # 计算损失
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # 反向传播
            losses.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # 统计
            total_loss += losses.item()
            for k in loss_dict.keys():
                if k in total_loss_dict:
                    total_loss_dict[k] += loss_dict[k].item()
                else:
                    total_loss_dict[k] = loss_dict[k].item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{losses.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(data_loader)
        avg_loss_dict = {k: v / len(data_loader) for k, v in total_loss_dict.items()}
        
        return avg_loss, avg_loss_dict

    def validate(self, data_loader, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_loss_dict = {}
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc='Validating'):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                outputs = self.model(torch.stack(images))
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                total_loss += losses.item()
                for k in loss_dict.keys():
                    if k in total_loss_dict:
                        total_loss_dict[k] += loss_dict[k].item()
                    else:
                        total_loss_dict[k] = loss_dict[k].item()
        
        avg_loss = total_loss / len(data_loader)
        avg_loss_dict = {k: v / len(data_loader) for k, v in total_loss_dict.items()}
        
        return avg_loss, avg_loss_dict

    def train(self, train_loader, val_loader, start_epoch, total_epochs, eval_interval, save_interval):
        """训练循环"""
        print(f"开始训练，共 {total_epochs} 个epoch")
        
        for epoch in range(start_epoch, total_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_loss_dict = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time
            
            # 记录训练损失
            self.train_losses.append({
                'epoch': epoch,
                'total_loss': train_loss,
                **train_loss_dict
            })
            
            print(f'Epoch {epoch}/{total_epochs} - Time: {epoch_time:.2f}s - Loss: {train_loss:.4f}')
            for k, v in train_loss_dict.items():
                print(f'  {k}: {v:.4f}')
                
            # 验证
            if val_loader is not None and (epoch % eval_interval == 0 or epoch == total_epochs - 1):
                val_loss, val_loss_dict = self.validate(val_loader, epoch)
                self.val_losses.append({
                    'epoch': epoch,
                    'total_loss': val_loss,
                    **val_loss_dict
                })
                print(f'验证集 Epoch {epoch} - Loss: {val_loss:.4f}')
            
            # 保存检查点
            if epoch % save_interval == 0 or epoch == total_epochs - 1:
                self.save_checkpoint(epoch, train_loss)
        
        # 保存训练记录
        self.save_training_log()

    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
        }
        
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def save_training_log(self):
        """保存训练日志"""
        log_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        log_path = os.path.join(self.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"训练日志已保存: {log_path}")


# 测试函数
def test_trainer():
    """测试训练器"""
    print("Testing trainer...")
    
    # 创建模拟数据
    from models.deformable_detr import DeformableDETR
    from models.criterion import SetCriterion, HungarianMatcher
    from util.scheduler import WarmupCosineSchedule
    
    class Config:
        backbone = 'resnet50'
        hidden_dim = 256
        num_queries = 100
        num_classes = 7
        nheads = 8
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 1024
        dropout = 0.1
    
    config = Config()
    
    # 创建模型
    model = DeformableDETR(config)
    
    # 创建损失函数
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(
        num_classes=7,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['labels', 'boxes']
    )
    
    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, total_steps=100)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device='cpu',
        output_dir='test_output'
    )
    
    print("✓ Trainer created successfully")
    return trainer


if __name__ == '__main__':
    test_trainer()