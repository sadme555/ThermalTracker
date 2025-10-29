# test_train.py
"""
训练系统测试 - 修复版本
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_training_components():
    """测试训练组件"""
    print("=" * 60)
    print("Testing Training Components")
    print("=" * 60)
    
    try:
        # 测试损失函数
        from models.criterion import HungarianMatcher, SetCriterion
        
        # 创建匹配器和损失函数
        matcher = HungarianMatcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        criterion = SetCriterion(7, matcher, weight_dict)
        
        print("✓ HungarianMatcher created")
        print("✓ SetCriterion created")
        
        # 创建测试数据
        outputs = {
            'pred_logits': torch.randn(2, 10, 8),  # [batch, queries, num_classes+1]
            'pred_boxes': torch.randn(2, 10, 4)    # [batch, queries, 4]
        }
        
        targets = [
            {
                'labels': torch.tensor([1, 3]),  # 2个目标
                'boxes': torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
            },
            {
                'labels': torch.tensor([2, 4, 6]),  # 3个目标
                'boxes': torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9], [0.1, 0.1, 0.2, 0.2]])
            }
        ]
        
        print("✓ Test data created")
        print(f"  Outputs: logits={outputs['pred_logits'].shape}, boxes={outputs['pred_boxes'].shape}")
        print(f"  Targets: {len(targets)} samples with {[len(t['labels']) for t in targets]} objects each")
        
        # 测试匈牙利匹配
        indices = matcher(outputs, targets)
        print("✓ Hungarian matching completed")
        for i, (idx1, idx2) in enumerate(indices):
            print(f"  Sample {i}: {len(idx1)} matches")
        
        # 测试损失计算
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())
        
        print("✓ Loss calculation completed")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler():
    """测试调度器"""
    print("\n" + "=" * 60)
    print("Testing Scheduler")
    print("=" * 60)
    
    try:
        from util.scheduler import WarmupCosineSchedule
        
        # 创建模拟优化器
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{'lr': 0.1}]
        
        optimizer = MockOptimizer()
        
        # 测试调度器
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=5, total_steps=20)
        
        print("Testing learning rate schedule:")
        lrs = []
        for step in range(1, 21):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            lrs.append(lr)
            if step in [1, 3, 5, 10, 15, 20]:
                print(f"  Step {step}: LR = {lr:.6f}")
        
        # 验证LR变化
        assert lrs[0] < lrs[4], "LR should increase during warmup"
        assert lrs[4] > lrs[-1], "LR should decrease after warmup"
        
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
        from models.deformable_detr import DeformableDETR
        from models.criterion import HungarianMatcher, SetCriterion
        from util.scheduler import WarmupCosineSchedule
        
        # 创建简单配置
        class Config:
            backbone = 'resnet50'
            hidden_dim = 128  # 小模型以加快测试
            num_queries = 10
            num_classes = 7
            nheads = 4
            num_encoder_layers = 2
            num_decoder_layers = 2
            dim_feedforward = 512
            dropout = 0.1
        
        config = Config()
        
        # 创建组件
        model = DeformableDETR(config)
        matcher = HungarianMatcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        criterion = SetCriterion(7, matcher, weight_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=10)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            output_dir='test_trainer_output'
        )
        
        print("✓ Trainer integration components created")
        
        # 测试检查点功能
        trainer.save_checkpoint(0)
        print("✓ Checkpoint saving works")
        
        # 测试加载
        trainer.load_checkpoint('test_trainer_output/checkpoint_epoch_0.pth')
        print("✓ Checkpoint loading works")
        
        # 清理
        import shutil
        if os.path.exists('test_trainer_output'):
            shutil.rmtree('test_trainer_output')
        
        print("✓ Trainer integration test passed!")
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
        from torch.utils.data import DataLoader, Subset
        
        # 获取配置
        config = get_default_config()
        
        # 构建迷你数据集（只取前10个样本）
        print("Building mini dataset...")
        full_dataset = build_infrared_dataset(config, is_train=True)
        mini_dataset = Subset(full_dataset, indices=range(min(10, len(full_dataset))))
        
        print(f"Mini dataset size: {len(mini_dataset)}")
        
        # 创建数据加载器
        data_loader = DataLoader(
            mini_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # 测试时设为0避免多进程问题
            collate_fn=collate_fn
        )
        
        print("✓ DataLoader created successfully")
        
        # 测试数据流
        for batch_idx, (images, targets) in enumerate(data_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images: {len(images)} tensors")
            print(f"  Targets: {len(targets)} dicts")
            
            for i, (img, target) in enumerate(zip(images, targets)):
                print(f"    Sample {i}: image={img.shape}, targets={len(target['boxes'])}")
            
            # 只测试一个批次
            break
        
        print("✓ Mini training data flow test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Mini training test failed: {e}")
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
        test_mini_training
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
        print("The training system is ready! You can now run full training.")
        return True
    else:
        print("\n❌ SOME TRAINING TESTS FAILED!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)