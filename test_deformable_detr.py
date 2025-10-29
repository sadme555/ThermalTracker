# test_deformable_detr_memory_safe.py
"""
内存安全的Deformable DETR测试
"""

import torch
import sys
import os
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_memory_safe():
    """内存安全测试"""
    print("=" * 60)
    print("Memory Safe Deformable DETR Test")
    print("=" * 60)
    
    try:
        from config import get_default_config
        from models.deformable_detr import DeformableDETR
        
        # 获取配置
        config = get_default_config()
        print("✓ Configuration loaded")
        
        # 进一步减少配置以确保内存安全
        config.model.num_queries = 30
        config.model.hidden_dim = 192
        config.model.num_encoder_layers = 2
        config.model.num_decoder_layers = 2
        
        print(f"  num_queries: {config.model.num_queries}")
        print(f"  hidden_dim: {config.model.hidden_dim}")
        print(f"  layers: {config.model.num_encoder_layers} encoder, {config.model.num_decoder_layers} decoder")
        
        # 创建模型
        model = DeformableDETR(config.model)
        print("✓ Model created with config")
        
        # 测试输入 - 使用小尺寸
        x = torch.randn(1, 3, 256, 320)
        print(f"Input shape: {x.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(x)
        
        print("✓ Forward pass successful")
        print(f"Output keys: {list(outputs.keys())}")
        
        # 验证输出格式
        assert 'pred_logits' in outputs, "Missing pred_logits"
        assert 'pred_boxes' in outputs, "Missing pred_boxes"
        
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        print(f"pred_logits: {pred_logits.shape}")
        print(f"pred_boxes: {pred_boxes.shape}")
        
        # 验证形状
        assert pred_logits.shape[0] == 1, f"Batch size should be 1, got {pred_logits.shape[0]}"
        assert pred_logits.shape[1] == config.model.num_queries, \
            f"Query number mismatch: expected {config.model.num_queries}, got {pred_logits.shape[1]}"
        assert pred_logits.shape[2] == config.model.num_classes + 1, \
            f"Class number mismatch: expected {config.model.num_classes + 1}, got {pred_logits.shape[2]}"
        
        assert pred_boxes.shape[0] == 1, f"Batch size should be 1, got {pred_boxes.shape[0]}"
        assert pred_boxes.shape[1] == config.model.num_queries, \
            f"Query number mismatch: expected {config.model.num_queries}, got {pred_boxes.shape[1]}"
        assert pred_boxes.shape[2] == 4, f"Should predict 4 box coordinates, got {pred_boxes.shape[2]}"
        
        # 清理内存
        del model, outputs, x
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ Config integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progressive_complexity():
    """渐进复杂度测试"""
    print("\n" + "=" * 60)
    print("Progressive Complexity Test")
    print("=" * 60)
    
    try:
        from config import get_default_config
        from models.deformable_detr import DeformableDETR
        
        # 测试不同复杂度的配置
        complexity_levels = [
            {'name': 'Level 1', 'queries': 20, 'dim': 128, 'layers': 2, 'size': (224, 288)},
            {'name': 'Level 2', 'queries': 40, 'dim': 192, 'layers': 3, 'size': (320, 400)},
            {'name': 'Level 3', 'queries': 60, 'dim': 256, 'layers': 4, 'size': (416, 512)},
        ]
        
        for level in complexity_levels:
            print(f"\n▶ Testing {level['name']}:")
            
            config = get_default_config()
            config.model.num_queries = level['queries']
            config.model.hidden_dim = level['dim']
            config.model.num_encoder_layers = level['layers']
            config.model.num_decoder_layers = level['layers']
            
            model = DeformableDETR(config.model)
            
            height, width = level['size']
            x = torch.randn(1, 3, height, width)
            
            with torch.no_grad():
                outputs = model(x)
            
            print(f"  Input: {x.shape}")
            print(f"  Output: logits={outputs['pred_logits'].shape}, boxes={outputs['pred_boxes'].shape}")
            
            # 验证
            assert outputs['pred_logits'].shape[1] == level['queries']
            assert outputs['pred_boxes'].shape[1] == level['queries']
            
            # 清理
            del model, outputs, x
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  ✓ {level['name']} passed")
        
        print("\n✓ All progressive complexity tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Progressive complexity test failed: {e}")
        return False


def main():
    """运行内存安全测试"""
    print("Running memory-safe Deformable DETR tests...")
    
    tests = [
        test_memory_safe,
        test_progressive_complexity
    ]
    
    results = []
    for test in tests:
        try:
            # 在每个测试前清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    if all(results):
        print("\n🎉 ALL MEMORY-SAFE TESTS PASSED!")
        print("The model works correctly with memory-optimized configurations.")
        print("You can now proceed to implement loss functions.")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Consider using even smaller configurations for your hardware.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)