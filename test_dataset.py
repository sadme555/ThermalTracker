import sys
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 在文件顶部导入所有需要的模块
try:
    from datasets.rgbt_tiny_config import RGBTinyConfig
    from datasets.infrared_small_target import InfraredSmallTargetDataset
    print("✓ 成功导入所有模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    # 继续执行，让各个测试函数处理具体的错误

print("当前工作目录:", os.getcwd())
print("目录内容:", os.listdir('.'))

def test_config():
    """测试配置模块"""
    try:
        config = RGBTinyConfig()
        print("✓ 成功创建配置实例")
        
        print(f"根路径: {config.root_path}")
        print(f"图像路径: {config.image_path}")
        print(f"标注路径: {config.annotation_path}")
        print(f"划分路径: {config.split_path}")
        
        # 检查路径是否存在
        print(f"根路径存在: {config.root_path.exists()}")
        print(f"图像路径存在: {config.image_path.exists()}")
        print(f"标注路径存在: {config.annotation_path.exists()}")
        print(f"划分路径存在: {config.split_path.exists()}")
        
        # 列出文件
        if config.annotation_path.exists():
            print("标注文件:", os.listdir(config.annotation_path))
            
        if config.split_path.exists():
            print("划分文件:", os.listdir(config.split_path))
            
        if config.image_path.exists():
            # 只显示前3个子目录
            subdirs = list(config.image_path.iterdir())[:3]
            print("图像子目录示例:", [d.name for d in subdirs])
            
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coco_annotations():
    """测试COCO标注文件"""
    try:
        from pycocotools.coco import COCO
        
        config = RGBTinyConfig()
        annotation_file = config.annotation_path / "instances_00_train2017.json"
        
        if not annotation_file.exists():
            print(f"✗ 标注文件不存在: {annotation_file}")
            return False
            
        print(f"✓ 加载COCO标注: {annotation_file}")
        coco = COCO(annotation_file)
        
        # 获取所有图像
        img_ids = coco.getImgIds()
        print(f"✓ 找到 {len(img_ids)} 张图像")
        
        # 获取所有类别
        cat_ids = coco.getCatIds()
        categories = coco.loadCats(cat_ids)
        print(f"✓ 找到 {len(categories)} 个类别:")
        for cat in categories:
            print(f"  - {cat['name']} (id: {cat['id']})")
            
        # 显示第一张图像的信息
        if len(img_ids) > 0:
            img_info = coco.loadImgs(img_ids[0])[0]
            print(f"✓ 第一张图像信息: {img_info}")
            
        return True
    except Exception as e:
        print(f"✗ COCO标注测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """测试数据集加载"""
    try:
        print("✓ 开始测试数据集加载...")
        dataset = InfraredSmallTargetDataset(split='train')
        
        print(f"✓ 数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试第一个样本
            image, target = dataset[0]
            print(f"✓ 成功加载第一个样本")
            print(f"  图像类型: {type(image)}")
            if isinstance(image, torch.Tensor):
                print(f"  图像形状: {image.shape}")
            else:
                print(f"  图像形状: {image.shape}")
                
            print(f"  目标键: {list(target.keys())}")
            print(f"  边界框数量: {len(target['boxes'])}")
            if len(target['boxes']) > 0:
                print(f"  第一个边界框: {target['boxes'][0]}")
            print(f"  标签: {target['labels']}")
            
        return True
    except Exception as e:
        print(f"✗ 数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_sample():
    """可视化样本"""
    try:
        dataset = InfraredSmallTargetDataset(split='train')
        
        if len(dataset) == 0:
            print("✗ 数据集为空")
            return False
            
        image, target = dataset[0]
        
        # 转换为numpy数组用于可视化
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = image
        
        # 创建图形
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)
        
        boxes = target['boxes']
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                if isinstance(box, torch.Tensor):
                    box = box.numpy()
                
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                
                label = target['labels'][i]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                
                ax.text(x1, y1, f'Class: {label}', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.6),
                        fontsize=8)
        
        plt.title(f"红外小目标检测样本 - 目标数量: {len(boxes)}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
        print("✓ 样本图像已保存为 dataset_sample.png")
        return True
        
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("开始测试红外小目标数据集")
    print("=" * 50)
    
    # 逐步测试
    tests = [
        ("配置测试", test_config),
        ("COCO标注测试", test_coco_annotations),
        ("数据集加载测试", test_dataset_loading),
        ("可视化测试", visualize_sample)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n▶ 执行 {test_name}...")
        if test_func():
            print(f"✓ {test_name} 通过!")
        else:
            print(f"✗ {test_name} 失败!")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过! 数据集准备就绪。")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
    print("=" * 50)