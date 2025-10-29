"""
测试完整模型的端到端流程
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.deformable_detr import test_deformable_detr
from config import ModelConfig

def test_end_to_end():
    """测试端到端流程"""
    print("=" * 60)
    print("测试端到端模型流程")
    print("=" * 60)
    
    # 使用真实配置
    config = ModelConfig(
        backbone='resnet50',
        hidden_dim=256,
        num_queries=100,
        num_classes=7,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        small_target_enhance=True
    )
    
    try:
        from models.deformable_detr import DeformableDETR
        
        print("▶ 使用真实配置构建模型...")
        model = DeformableDETR(config)
        
        # 测试不同批次的输入
        batch_sizes = [1, 2, 4]
        image_sizes = [(512, 640), (384, 480)]  # 红外图像常见尺寸
        
        for batch_size in batch_sizes:
            for img_size in image_sizes:
                print(f"\n▶ 测试批次大小: {batch_size}, 图像尺寸: {img_size}")
                
                # 创建测试输入
                x = torch.randn(batch_size, 3, img_size[0], img_size[1])
                
                # 前向传播
                with torch.no_grad():
                    outputs = model(x)
                
                print(f"  ✓ 输入: {x.shape}")
                print(f"  ✓ 输出logits: {outputs['pred_logits'].shape}")
                print(f"  ✓ 输出boxes: {outputs['pred_boxes'].shape}")
                
                # 验证输出范围
                assert outputs['pred_boxes'].min() >= 0, "边界框坐标应 >= 0"
                assert outputs['pred_boxes'].max() <= 1, "边界框坐标应 <= 1"
                print(f"  ✓ 边界框坐标范围正确: [0, 1]")
        
        print("\n🎉 所有端到端测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试内存使用"""
    if not torch.cuda.is_available():
        print("未检测到GPU，跳过内存测试")
        return
    
    print("\n" + "=" * 60)
    print("测试GPU内存使用")
    print("=" * 60)
    
    config = ModelConfig(
        backbone='resnet50',
        hidden_dim=256,
        num_queries=100,
        num_classes=7
    )
    
    from models.deformable_detr import DeformableDETR
    
    model = DeformableDETR(config).cuda()
    
    # 测试不同配置的内存使用
    test_cases = [
        (1, 512, 640),  # 小批次，标准尺寸
        (2, 512, 640),  # 中等批次
        (4, 384, 480),  # 较大批次，较小尺寸
    ]
    
    for batch_size, height, width in test_cases:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_memory = torch.cuda.memory_allocated() / 1024**3
        
        # 前向传播
        x = torch.randn(batch_size, 3, height, width).cuda()
        with torch.no_grad():
            outputs = model(x)
        
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated() / 1024**3
        memory_used = end_memory - start_memory
        
        print(f"批次 {batch_size}, 尺寸 {height}x{width}: {memory_used:.2f} GB")
        
        if memory_used > 4.0:
            print("  ⚠ 内存使用较高")
        else:
            print("  ✓ 内存使用合理")

if __name__ == '__main__':
    # 运行基础模型测试
    print("运行基础模型测试...")
    model, outputs = test_deformable_detr()
    
    if model is not None:
        # 运行端到端测试
        success = test_end_to_end()
        
        # 运行内存测试
        test_memory_usage()
        
        if success:
            print("\n🎉 所有模型测试完成! 可以开始训练了。")
        else:
            print("\n❌ 模型测试失败，请检查实现。")
    else:
        print("\n❌ 基础模型测试失败，无法继续。")