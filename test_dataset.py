# test_dataset.py
"""
红外小目标数据集测试脚本 - 修复版本
"""

import os
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """设置环境并检查依赖"""
    print("✓ 成功导入所有模块")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"目录内容: {os.listdir('.')}")
    return True

def test_config():
    """测试配置系统"""
    print("\n▶ 执行 配置测试...")
    try:
        from config import get_default_config
        config = get_default_config()
        print("✓ 成功创建配置实例")
        
        # 检查数据集路径
        dataset_cfg = config.data.dataset
        print(f"根路径: {dataset_cfg.root_path}")
        print(f"图像路径: {dataset_cfg.image_path}")
        print(f"标注路径: {dataset_cfg.annotation_path}")
        print(f"划分路径: {dataset_cfg.split_path}")
        
        # 检查路径是否存在
        print(f"根路径存在: {dataset_cfg.root_path.exists()}")
        print(f"图像路径存在: {dataset_cfg.image_path.exists()}")
        print(f"标注路径存在: {dataset_cfg.annotation_path.exists()}")
        print(f"划分路径存在: {dataset_cfg.split_path.exists()}")
        
        # 检查文件
        if dataset_cfg.annotation_path.exists():
            annotation_files = list(dataset_cfg.annotation_path.glob("*.json"))
            print(f"标注文件: {[f.name for f in annotation_files]}")
        
        if dataset_cfg.split_path.exists():
            split_files = list(dataset_cfg.split_path.glob("*.txt"))
            print(f"划分文件: {[f.name for f in split_files]}")
        
        if dataset_cfg.image_path.exists():
            image_dirs = list(dataset_cfg.image_path.iterdir())
            print(f"图像子目录示例: {[d.name for d in image_dirs[:3]]}")
        
        print("✓ 配置测试 通过!")
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_coco_annotations():
    """测试COCO标注文件"""
    print("\n▶ 执行 COCO标注测试...")
    try:
        from pycocotools.coco import COCO
        from config import get_default_config
        
        config = get_default_config()
        dataset_cfg = config.data.dataset
        
        # 测试训练标注文件
        train_annotation_path = dataset_cfg.get_annotation_path('train')
        print(f"✓ 加载COCO标注: {train_annotation_path}")
        
        coco = COCO(train_annotation_path)
        
        # 获取图像和类别信息
        image_ids = coco.getImgIds()
        category_ids = coco.getCatIds()
        categories = coco.loadCats(category_ids)
        
        print(f"✓ 找到 {len(image_ids)} 张图像")
        print(f"✓ 找到 {len(categories)} 个类别:")
        for cat in categories:
            print(f"  - {cat['name']} (id: {cat['id']})")
        
        # 检查第一张图像
        if len(image_ids) > 0:
            img_info = coco.loadImgs(image_ids[0])[0]
            print(f"✓ 第一张图像信息: {img_info}")
        
        print("✓ COCO标注测试 通过!")
        return True
        
    except Exception as e:
        print(f"✗ COCO标注测试失败: {e}")
        return False

def test_dataset_loading():
    """测试数据集加载 - 修复版本"""
    print("\n▶ 执行 数据集加载测试...")
    try:
        from config import get_default_config
        from datasets.infrared_small_target import build_infrared_dataset
        
        print("✓ 开始测试数据集加载...")
        
        # 获取配置
        config = get_default_config()
        
        # 检查数据集路径
        dataset_cfg = config.data.dataset
        print("检查数据集路径...")
        print(f"root路径: {dataset_cfg.root_path} - 存在: {dataset_cfg.root_path.exists()}")
        if dataset_cfg.root_path.exists():
            print(f"  内容: {[item.name for item in dataset_cfg.root_path.iterdir()]}")
        
        print(f"images路径: {dataset_cfg.image_path} - 存在: {dataset_cfg.image_path.exists()}")
        if dataset_cfg.image_path.exists():
            print(f"  内容: {[item.name for item in list(dataset_cfg.image_path.iterdir())[:3]]}")
        
        print(f"annotations路径: {dataset_cfg.annotation_path} - 存在: {dataset_cfg.annotation_path.exists()}")
        if dataset_cfg.annotation_path.exists():
            print(f"  内容: {[item.name for item in list(dataset_cfg.annotation_path.iterdir())[:3]]}")
        
        print(f"splits路径: {dataset_cfg.split_path} - 存在: {dataset_cfg.split_path.exists()}")
        if dataset_cfg.split_path.exists():
            print(f"  内容: {[item.name for item in list(dataset_cfg.split_path.iterdir())[:3]]}")
        
        # 使用正确的函数构建数据集
        train_annotation_path = dataset_cfg.get_annotation_path('train')
        print(f"加载标注文件: {train_annotation_path}")
        
        # 构建数据集
        dataset = build_infrared_dataset(config, is_train=True)
        
        print(f"数据集加载成功: {len(dataset)} 张图像")
        
        # 获取类别信息
        if hasattr(dataset, 'class_names'):
            print(f"类别信息: {dataset.class_names}")
        elif hasattr(dataset, 'categories'):
            print(f"类别信息: {[cat['name'] for cat in dataset.categories]}")
        
        print(f"✓ 数据集大小: {len(dataset)}")
        
        # 测试加载第一个样本
        if len(dataset) > 0:
            print(f"加载图像 0")
            img, target = dataset[0]
            
            print(f"✓ 成功加载第一个样本")
            print(f"  图像类型: {type(img)}")
            print(f"  图像形状: {img.shape}")
            print(f"  目标键: {list(target.keys())}")
            print(f"  边界框数量: {len(target['boxes'])}")
            
            if len(target['boxes']) > 0:
                print(f"  第一个边界框: {target['boxes'][0]}")
                print(f"  标签: {target['labels']}")
        
        print("✓ 数据集加载测试 通过!")
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_sample():
    """可视化样本 - 修复版本"""
    print("\n▶ 执行 可视化测试...")
    try:
        from config import get_default_config
        from datasets.infrared_small_target import build_infrared_dataset
        
        print("检查数据集路径...")
        config = get_default_config()
        dataset_cfg = config.data.dataset
        
        print(f"root路径: {dataset_cfg.root_path} - 存在: {dataset_cfg.root_path.exists()}")
        if dataset_cfg.root_path.exists():
            print(f"  内容: {[item.name for item in dataset_cfg.root_path.iterdir()]}")
        
        print(f"images路径: {dataset_cfg.image_path} - 存在: {dataset_cfg.image_path.exists()}")
        if dataset_cfg.image_path.exists():
            print(f"  内容: {[item.name for item in list(dataset_cfg.image_path.iterdir())[:3]]}")
        
        print(f"annotations路径: {dataset_cfg.annotation_path} - 存在: {dataset_cfg.annotation_path.exists()}")
        if dataset_cfg.annotation_path.exists():
            print(f"  内容: {[item.name for item in list(dataset_cfg.annotation_path.iterdir())[:3]]}")
        
        print(f"splits路径: {dataset_cfg.split_path} - 存在: {dataset_cfg.split_path.exists()}")
        if dataset_cfg.split_path.exists():
            print(f"  内容: {[item.name for item in list(dataset_cfg.split_path.iterdir())[:3]]}")
        
        # 使用正确的函数构建数据集
        train_annotation_path = dataset_cfg.get_annotation_path('train')
        print(f"加载标注文件: {train_annotation_path}")
        
        dataset = build_infrared_dataset(config, is_train=True)
        
        print(f"数据集加载成功: {len(dataset)} 张图像")
        
        if hasattr(dataset, 'class_names'):
            print(f"类别信息: {dataset.class_names}")
        elif hasattr(dataset, 'categories'):
            print(f"类别信息: {[cat['name'] for cat in dataset.categories]}")
        
        # 可视化第一个样本
        if len(dataset) > 0:
            print(f"加载图像 0")
            img, target = dataset[0]
            
            # 转换为numpy用于可视化
            if torch.is_tensor(img):
                img_np = img.permute(1, 2, 0).numpy()
                # 反标准化
                mean = np.array(config.data.image_mean)
                std = np.array(config.data.image_std)
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img
            
            # 创建可视化
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_np)
            
            # 绘制边界框
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                labels = target['labels']
                
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    if torch.is_tensor(box):
                        box = box.numpy()
                    if torch.is_tensor(label):
                        label = label.item()
                    
                    # 边界框坐标 [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 创建矩形框
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加标签
                    class_name = f"class_{label}"
                    if hasattr(dataset, 'class_names') and label < len(dataset.class_names):
                        class_name = dataset.class_names[label]
                    
                    ax.text(x1, y1 - 5, class_name, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                           color='white', fontsize=8)
            
            ax.set_title('红外小目标检测样本', fontsize=16, pad=20)
            ax.axis('off')
            plt.tight_layout()
            
            # 保存图像
            plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
            print("✓ 样本图像已保存为 dataset_sample.png")
            
            # 显示图像
            plt.show()
        
        print("✓ 可视化测试 通过!")
        return True
        
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("开始测试红外小目标数据集")
    print("=" * 50)
    
    # 设置环境
    if not setup_environment():
        return
    
    # 运行测试
    tests = [
        test_config,
        test_coco_annotations,
        test_dataset_loading,
        visualize_sample
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试 {test.__name__} 执行失败: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    
    # 输出结果
    if all(results):
        print("🎉 所有测试通过! 数据集准备就绪。")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
    
    print("=" * 50)

if __name__ == '__main__':
    main()