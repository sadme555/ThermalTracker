#!/usr/bin/env python3
"""
ThermalTracker 主训练脚本 - 完整修复版本
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import build_infrared_dataset, collate_fn
from models import DeformableDETR, SetCriterion, HungarianMatcher
from engine import train_one_epoch, evaluate, EarlyStopping
from util import create_optimizer, create_scheduler, save_checkpoint, load_checkpoint
from util.motdet_eval import InfraredSmallTargetEvaluator
from config import get_config, save_config


def build_model(config):
    """构建模型"""
    print("Building model...")
    
    # 创建模型配置
    class ModelConfig:
        backbone = config.model.backbone
        hidden_dim = config.model.hidden_dim
        num_queries = config.model.num_queries
        num_classes = config.data.num_classes
        nheads = config.model.nheads
        num_encoder_layers = config.model.num_encoder_layers
        num_decoder_layers = config.model.num_decoder_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.dropout
        num_feature_levels = config.model.num_feature_levels
    
    model_config = ModelConfig()
    model = DeformableDETR(model_config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def build_criterion(config):
    """构建损失函数"""
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
    
    return criterion


def build_datasets(config):
    """构建数据集"""
    print("Building datasets...")
    
    # 确保数据集路径存在
    if not os.path.exists(config.data.dataset_root):
        print(f"警告: 数据集路径不存在: {config.data.dataset_root}")
        print("请确保已正确设置数据集路径")
    
    try:
        train_dataset = build_infrared_dataset(config, is_train=True)
        val_dataset = build_infrared_dataset(config, is_train=False)
    except Exception as e:
        print(f"数据集构建错误: {e}")
        print("尝试使用备用方法...")
        # 使用备用方法
        train_dataset, val_dataset = build_datasets_fallback(config)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def build_datasets_fallback(config):
    """备用数据集构建方法"""
    from datasets.infrared_small_target import InfraredSmallTargetDataset
    
    # 创建简化的配置
    class SimpleConfig:
        class data:
            dataset_root = config.data.dataset_root
            image_size = getattr(config.data, 'image_size', (512, 640))
            use_augmentation = True
            horizontal_flip_prob = 0.5
            small_target_augmentation = True
            min_scale = 0.8
            max_scale = 1.2
            color_jitter_prob = 0.5
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            num_classes = getattr(config.data, 'num_classes', 7)
        
        data = data()
    
    train_dataset = InfraredSmallTargetDataset(SimpleConfig, is_train=True)
    val_dataset = InfraredSmallTargetDataset(SimpleConfig, is_train=False)
    
    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, config):
    """构建数据加载器"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train(config):
    """主训练函数 - 最终修复版本"""
    print("=" * 60)
    print("ThermalTracker Training Started")
    print("=" * 60)
    
    # 设置设备
    device = torch.device(config.training.device)
    print(f"Using device: {device}")
    
    # 构建组件
    train_dataset, val_dataset = build_datasets(config)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, config)
    model = build_model(config)
    criterion = build_criterion(config)
    
    # 移动到设备
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 修复：使用正确的属性名 lr
    lr = getattr(config.training, 'lr', getattr(config.training, 'learning_rate', 1e-5))
    print(f"Using learning rate: {lr}")
    
    # 优化器和调度器
    optimizer = create_optimizer(
        model, 
        lr, 
        config.training.weight_decay
    )
    
    scheduler = create_scheduler(optimizer, config)
    
    # 早停机制 - 修复：正确初始化
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta
    )
    
    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)
    save_config(config, os.path.join(config.training.output_dir, 'config.yaml'))
    
    # 训练循环 - 修复：正确使用早停机制
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        print("-" * 50)
        
        try:
            # 训练一个epoch
            train_stats = train_one_epoch(
                model, criterion, train_loader, optimizer, 
                device, epoch, config.training.clip_max_norm
            )
            
            # 验证
            val_stats = evaluate(model, criterion, val_loader, device)
            val_loss = val_stats['loss']
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config.training.output_dir, 'best_model.pth')
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config, best_model_path
                )
                print(f"Best model saved with val_loss: {val_loss:.4f}")
                # 保存最佳模型状态用于早停
                early_stopping.save_checkpoint(model)
            
            # 定期保存检查点
            if (epoch + 1) % config.training.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config.training.output_dir, f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config, checkpoint_path
                )
            
            # 早停检查 - 修复：正确调用
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"Status: {early_stopping.status}")
                break
            else:
                print(f"Status: {early_stopping.status}")
                
        except Exception as e:
            print(f"❌ Error in epoch {epoch + 1}: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果连续3个epoch都出错，停止训练
            if epoch > 0 and epoch % 3 == 0:
                print("Too many consecutive errors, stopping training")
                break
            continue
    
    # 恢复最佳模型权重 - 修复：正确调用
    early_stopping.restore_best_weights(model)
    
    # 保存最终模型
    final_model_path = os.path.join(config.training.output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, scheduler, epoch, config, final_model_path)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {config.training.output_dir}")


def evaluate_model(config):
    """模型评估"""
    print("Evaluating model...")
    
    device = torch.device(config.training.device)
    
    # 构建模型
    model = build_model(config)
    model = model.to(device)
    
    # 加载训练好的权重
    checkpoint_path = config.evaluation.checkpoint_path
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    load_checkpoint(model, None, None, checkpoint_path, device)
    model.eval()
    
    # 构建数据集
    _, val_dataset = build_datasets(config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn
    )
    
    # 评估器
    evaluator = InfraredSmallTargetEvaluator()
    
    print("Starting evaluation...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
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
    
    # 计算评估指标
    metrics = evaluator.evaluate(all_predictions, all_targets)
    
    print("\nEvaluation Results:")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ThermalTracker Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                       help='运行模式: train 或 eval')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='',
                       help='设备 (cuda, cuda:0, cpu)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.config)
    
    # 覆盖配置参数
    if args.resume:
        config.training.resume = args.resume
    if args.device:
        config.training.device = args.device
    
    # 设置随机种子
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)
    
    # 运行模式
    if args.mode == 'train':
        train(config)
    else:
        evaluate_model(config)


if __name__ == '__main__':
    main()