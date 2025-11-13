# test_train.py
"""
训练组件测试 - 修复版本
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_training_components():
    """测试训练组件"""
    print("=" * 60)
    print("Testing Training Components")
    print("=" * 60)
    
    try:
        from models.criterion import HungarianMatcher, SetCriterion
        
        # 测试匈牙利匹配器
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0, 
            cost_giou=2.0
        )
        print("✓ HungarianMatcher created")
        
        # 测试损失函数
        weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
        
        criterion = SetCriterion(
            num_classes=7,
            matcher=matcher,
            weight_dict=weight_dict
        )
        print("✓ SetCriterion created")
        
        # 测试损失计算
        outputs = {
            'pred_logits': torch.randn(2, 100, 8),  # [batch, queries, classes+1]
            'pred_boxes': torch.rand(2, 100, 4)     # [batch, queries, 4]
        }
        
        targets = [
            {
                'labels': torch.tensor([0, 1, 2]),
                'boxes': torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]])
            },
            {
                'labels': torch.tensor([3, 4]),
                'boxes': torch.tensor([[0.2, 0.2, 0.3, 0.3], [0.6, 0.6, 0.7, 0.7]])
            }
        ]
        
        losses = criterion(outputs, targets)
        print("✓ Loss computation test passed")
        print(f"  Loss keys: {list(losses.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler():
    """测试学习率调度器"""
    print("\n" + "=" * 60)
    print("Testing Scheduler")
    print("=" * 60)
    
    try:
        from util.scheduler import WarmupCosineSchedule
        
        # 创建简单的优化器
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # 创建调度器
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=5,
            total_steps=20,
            min_lr=0.0
        )
        
        # 测试调度器
        print("Testing learning rate schedule:")
        lrs = []
        for step in range(21):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if step in [1, 3, 5, 10, 15, 20]:
                print(f"  Step {step}: LR = {lr:.6f}")
        
        # 验证学习率变化
        assert lrs[0] > 0, "LR should be positive"
        assert lrs[5] > lrs[0], "LR should increase during warmup"
        assert lrs[20] == 0.0, "LR should reach min_lr at the end"
        
        print("✓ Scheduler test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Scheduler test failed: {e}")
        return False


def test_trainer_integration():
    """测试训练器集成"""
    print("\n" + "=" * 60)
    print("Testing Trainer Integration")
    print("=" * 60)
    
    try:
        from engine.trainer import Trainer
        from models.criterion import HungarianMatcher, SetCriterion
        
        # 创建简单的模型和组件
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
            
            def forward(self, x):
                return {'pred_logits': self.linear(x), 'pred_boxes': torch.rand(1, 10, 4)}
        
        model = SimpleModel()
        
        # 创建损失函数
        matcher = HungarianMatcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        criterion = SetCriterion(2, matcher, weight_dict)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建调度器
        from util.scheduler import WarmupCosineSchedule
        scheduler = WarmupCosineSchedule(optimizer, 10, 100)
        
        # 创建训练器
        device = torch.device('cpu')
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            device=device,
            output_dir='./test_output'
        )
        
        print("✓ Trainer import successful")
        print(f"Trainer initialized on device: {device}")
        print("✓ Trainer creation successful")
        print(f"  Device: {device}")
        print(f"  Output dir: ./test_output")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """测试迷你训练流程"""
    print("\n" + "=" * 60)
    print("Testing Mini Training")
    print("=" * 60)
    
    try:
        from config import get_default_config
        from datasets.infrared_small_target import build_infrared_dataset, collate_fn
        from torch.utils.data import DataLoader
        
        # 获取配置
        config = get_default_config()
        config.data.batch_size = 2
        
        # 构建数据集
        dataset = build_infrared_dataset(config, is_train=True)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,  # 测试时使用0个worker
            collate_fn=collate_fn
        )
        
        print("✓ DataLoader created successfully")
        
        # 测试一个批次
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 1:  # 只测试第一个批次
                break
                
            print(f"Batch {batch_idx}:")
            print(f"  Images: {len(images)} tensors")
            print(f"  Targets: {len(targets)} dicts")
            
            for i, (img, target) in enumerate(zip(images, targets)):
                print(f"    Sample {i}: image={img.shape}, targets={len(target['boxes'])}")
        
        print("✓ Mini training data flow test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_training_flow():
    """测试模型训练流程"""
    print("\n" + "=" * 60)
    print("Testing Model Training Flow")
    print("=" * 60)
    
    try:
        from config import get_default_config
        from models.deformable_detr import DeformableDETR
        from models.criterion import HungarianMatcher, SetCriterion
        from datasets.infrared_small_target import build_infrared_dataset, collate_fn
        from torch.utils.data import DataLoader
        
        # 获取配置
        config = get_default_config()
        config.data.batch_size = 2
        
        # 创建模型
        model = DeformableDETR(config.model)
        print("✓ Model created for training test")
        
        # 创建损失函数
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        )
        weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 5.0, 
            'loss_giou': 2.0
        }
        criterion = SetCriterion(
            num_classes=config.model.num_classes,
            matcher=matcher,
            weight_dict=weight_dict
        )
        print("✓ Criterion created for training test")
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
        print("✓ Optimizer created")
        
        # 创建数据加载器
        dataset = build_infrared_dataset(config, is_train=True)
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # 测试一个训练步骤
        model.train()
        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx >= 1:  # 只测试一个批次
                break
                
            # 前向传播
            outputs = model(images[0].unsqueeze(0))  # 测试单个样本
            
            # 计算损失
            single_target = [targets[0]]  # 单个目标
            losses = criterion(outputs, single_target)
            
            # 反向传播
            total_loss = sum(losses.values())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print("✓ Training step completed successfully")
            print(f"  Total loss: {total_loss.item():.4f}")
            print(f"  Individual losses: { {k: v.item() for k, v in losses.items()} }")
        
        print("✓ Model training flow test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model training flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有训练测试"""
    print("Running training tests...")
    
    tests = [
        test_training_components,
        test_scheduler,
        test_trainer_integration,
        test_mini_training,
        test_model_training_flow
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TRAINING TEST RESULTS:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    if all(results):
        print("\n🎉 ALL TRAINING TESTS PASSED!")
        print("The training pipeline is ready for use!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above before proceeding with training.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)