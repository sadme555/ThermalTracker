# util/scheduler.py
"""
学习率调度器
"""

import math
from torch.optim import Optimizer


class WarmupCosineSchedule:
    """带warmup的余弦退火调度器"""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self.get_lr()
        
        for param_group, new_lr in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = new_lr
    
    def get_lr(self):
        """计算当前学习率"""
        if self.current_step < self.warmup_steps:
            # Warmup阶段：线性增加
            progress = self.current_step / self.warmup_steps
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_mult = self.min_lr + (1.0 - self.min_lr) * lr_mult
            
            return [base_lr * lr_mult for base_lr in self.base_lrs]
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']