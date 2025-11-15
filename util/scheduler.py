"""
优化器和学习率调度器 - 修复参数问题版本
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
import math


def create_optimizer(model, lr, weight_decay):
    """
    创建优化器 - 修复类型问题
    """
    # 确保参数是浮点数
    try:
        lr = float(lr)
        weight_decay = float(weight_decay)
    except (ValueError, TypeError) as e:
        print(f"Error converting lr/weight_decay to float: {e}")
        print(f"lr: {lr}, type: {type(lr)}")
        print(f"weight_decay: {weight_decay}, type: {type(weight_decay)}")
        # 使用默认值
        lr = 1e-4
        weight_decay = 1e-4
    
    # 参数分组
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad],
            "lr": lr
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": lr * 0.1
        },
    ]
    
    print(f"Creating optimizer with lr={lr}, weight_decay={weight_decay}")
    
    # 创建优化器
    optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    
    return optimizer


def create_scheduler(optimizer, config, num_epochs=None):
    """
    创建学习率调度器 - 修复参数问题
    """
    # 如果没有提供num_epochs，从config中获取
    if num_epochs is None:
        num_epochs = getattr(config.training, 'epochs', 100)
    
    scheduler_type = getattr(config.training, 'scheduler', 'step')
    
    if scheduler_type == 'step':
        # 步长调度器
        step_size = getattr(config.training, 'lr_step', 30)
        gamma = getattr(config.training, 'lr_gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_type == 'multi_step':
        # 多步长调度器
        milestones = getattr(config.training, 'lr_milestones', [30, 60])
        gamma = getattr(config.training, 'lr_gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
    elif scheduler_type == 'cosine':
        # 余弦退火调度器
        T_max = num_epochs
        eta_min = getattr(config.training, 'min_lr', 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
    elif scheduler_type == 'warmup_cosine':
        # 带热身的余弦退火
        T_max = num_epochs - getattr(config.training, 'warmup_epochs', 5)
        eta_min = getattr(config.training, 'min_lr', 1e-6)
        base_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        warmup_epochs = getattr(config.training, 'warmup_epochs', 5)
        scheduler = WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
        
    else:
        # 默认：恒定学习率
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    print(f"Created {scheduler_type} scheduler for {num_epochs} epochs")
    return scheduler


class WarmupScheduler:
    """
    带热身的学习率调度器
    """
    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        
        # 存储初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 热身阶段：线性增加学习率
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # 正常调度
            self.base_scheduler.step()
            
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.base_lrs = state_dict['base_lrs']


def create_warmup_scheduler(optimizer, warmup_epochs, config, num_epochs=None):
    """
    创建带热身的调度器
    """
    base_scheduler = create_scheduler(optimizer, config, num_epochs)
    return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)


# 测试函数
def test_scheduler():
    """测试调度器"""
    print("Testing scheduler...")
    
    # 创建模拟模型
    class MockModel:
        def named_parameters(self):
            return [('backbone.weight', torch.randn(10, 10, requires_grad=True)),
                   ('head.weight', torch.randn(5, 10, requires_grad=True))]
    
    model = MockModel()
    
    # 测试优化器创建
    optimizer = create_optimizer(model, 1e-4, 1e-4)
    print(f"Optimizer: {type(optimizer)}")
    print(f"Learning rates: {[group['lr'] for group in optimizer.param_groups]}")
    
    # 测试调度器创建
    class MockConfig:
        class training:
            scheduler = 'step'
            lr_step = 10
            lr_gamma = 0.1
            epochs = 100
    
    config = MockConfig()
    
    # 测试不同的调用方式
    scheduler1 = create_scheduler(optimizer, config)
    print(f"Scheduler 1: {type(scheduler1)}")
    
    scheduler2 = create_scheduler(optimizer, config, num_epochs=50)
    print(f"Scheduler 2: {type(scheduler2)}")
    
    print("✓ Scheduler test passed!")


if __name__ == '__main__':
    test_scheduler()