# engine/trainer.py
"""
训练器实现 - 修复版本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import math


class Trainer:
    """
    训练器类 - 管理整个训练流程
    """
    
    def __init__(self, model, criterion, optimizer, lr_scheduler, device, output_dir, 
                 distributed=False, rank=0, clip_max_norm=0.1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.output_dir = output_dir
        self.distributed = distributed
        self.rank = rank
        self.clip_max_norm = clip_max_norm
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练统计
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, data_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_loss_dict = {}
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}') if self.rank == 0 else data_loader
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # 移动到设备
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
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
            
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
            if self.rank == 0:
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
        
        if self.rank == 0:
            print(f'Validation Epoch {epoch} - Loss: {avg_loss:.4f}')
            for k, v in avg_loss_dict.items():
                print(f'  {k}: {v:.4f}')
        
        return avg_loss, avg_loss_dict

    def train(self, train_loader, val_loader=None, start_epoch=0, total_epochs=50, 
              eval_interval=5, save_interval=10):
        """主训练循环"""
        print(f"开始训练，共 {total_epochs} 个epoch")
        
        for epoch in range(start_epoch, total_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_loss_dict = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time
            
            if self.rank == 0:
                print(f'Epoch {epoch}/{total_epochs} - Time: {epoch_time:.2f}s - Loss: {train_loss:.4f}')
                for k, v in train_loss_dict.items():
                    print(f'  {k}: {v:.4f}')
                
                # 保存训练损失
                self.train_losses.append(train_loss)
                
                # 保存检查点
                if epoch % save_interval == 0 or epoch == total_epochs - 1:
                    self.save_checkpoint(epoch, train_loss)
                
                # 验证
                if val_loader is not None and epoch % eval_interval == 0:
                    val_loss, val_loss_dict = self.validate(val_loader, epoch)
                    self.val_losses.append(val_loss)
                    
                    # 保存最佳模型
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(epoch, train_loss, is_best=True)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"检查点已加载: {checkpoint_path}")
        return checkpoint['epoch']