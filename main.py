# main.py
"""
红外小目标检测主训练脚本 - 完整版本
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import get_default_config, print_config
from datasets.infrared_small_target import build_infrared_dataset, collate_fn
from models.deformable_detr import DeformableDETR
from models.criterion import SetCriterion, HungarianMatcher
from engine.trainer import Trainer
from util.scheduler import WarmupCosineSchedule


def setup_config(args):
    """设置配置"""
    config = get_default_config()
    
    # 更新配置参数
    config.model.backbone = args.backbone
    config.model.hidden_dim = args.hidden_dim
    config.model.num_queries = args.num_queries
    config.data.batch_size = args.batch_size
    config.data.dataset_root = args.dataset_root
    config.train.lr = args.lr
    config.train.epochs = args.epochs
    config.train.weight_decay = args.weight_decay
    config.experiment.output_dir = args.output_dir
    config.experiment.device = args.device
    
    return config


def build_model(config):
    """构建模型"""
    print("构建Deformable DETR模型...")
    
    model = DeformableDETR(config.model)
    return model


def build_criterion(config):
    """构建损失函数"""
    print("构建损失函数...")
    
    # 匈牙利匹配器
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )
    
    # 损失权重
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    }
    
    # 损失函数
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(
        num_classes=config.model.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=losses
    )
    
    return criterion


def build_dataloaders(config):
    """构建数据加载器"""
    print("构建数据加载器...")
    
    train_dataset = build_infrared_dataset(config, is_train=True)
    val_dataset = build_infrared_dataset(config, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory
    )
    
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    return train_loader, val_loader


def build_optimizer(config, model):
    """构建优化器"""
    
    # 参数分组：骨干网络和其他参数
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad],
            "lr": config.train.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": config.train.lr * 0.1,  # 骨干网络学习率更低
        },
    ]
    
    optimizer = optim.AdamW(
        param_dicts, 
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )
    
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='红外小目标检测训练')
    parser.add_argument('--backbone', default='resnet50', help='骨干网络')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏维度')
    parser.add_argument('--num_queries', type=int, default=100, help='查询数量')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--dataset_root', default='datasets/RGBT-Tiny', help='数据集根路径')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--output_dir', default='outputs', help='输出目录')
    parser.add_argument('--device', default='cuda', help='训练设备')
    
    args = parser.parse_args()
    
    # 设置配置
    config = setup_config(args)
    print_config(config)
    
    # 设置设备
    device = torch.device(config.experiment.device)
    print(f"使用设备: {device}")
    
    # 构建数据加载器
    train_loader, val_loader = build_dataloaders(config)
    
    # 构建模型
    model = build_model(config)
    model = model.to(device)
    
    # 构建损失函数
    criterion = build_criterion(config)
    criterion = criterion.to(device)
    
    # 构建优化器
    optimizer = build_optimizer(config, model)
    
    # 构建学习率调度器
    total_steps = len(train_loader) * config.train.epochs
    warmup_steps = len(train_loader) * 5  # 5个epoch的warmup
    lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)
    
    # 构建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        output_dir=config.experiment.output_dir
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型信息:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 开始训练
    print(f"\n开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=0,
        total_epochs=config.train.epochs,
        eval_interval=5,
        save_interval=10
    )
    
    print(f"\n🎉 训练完成! 模型保存在: {config.experiment.output_dir}")


if __name__ == '__main__':
    main()