# datasets/infrared_small_target.py
"""
红外小目标数据集加载器 - 修复导入问题
"""

import torch
import torch.utils.data
from pycocotools.coco import COCO
import os
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple
import numpy as np

# 导入配置
try:
    from config import Config
except ImportError:
    # 如果config.py不存在，使用默认配置
    from dataclasses import dataclass
    @dataclass
    class Config:
        data = type('DataConfig', (), {
            'dataset': type('DatasetConfig', (), {
                'dataset_root': 'datasets/RGBT-Tiny',
                'image_dir': 'images', 
                'annotation_dir': 'annotations_coco',
                'train_annotations': ['instances_00_train2017.json'],
                'val_annotations': ['instances_00_test2017.json'],
                'image_size': (512, 640)
            }),
            'batch_size': 4,
            'num_workers': 4
        })()


class InfraredSmallTargetDataset(torch.utils.data.Dataset):
    def __init__(self, config: Config, is_train: bool = True):
        """
        初始化数据集
        
        Args:
            config: 配置对象
            is_train: 是否为训练模式
        """
        self.config = config
        self.is_train = is_train
        
        # 使用现有的RGBTinyConfig
        self.dataset_cfg = config.data.dataset
        
        # 获取标注文件路径
        split = 'train' if is_train else 'test'
        self.annotation_path = self.dataset_cfg.get_annotation_path(split)
        
        # 加载COCO标注
        self.coco = COCO(self.annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # 类别信息
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.class_names = [cat['name'] for cat in self.categories]
        self.num_classes = len(self.class_names)
        
        # 数据增强
        self.transforms = self._build_transforms()
        
        print(f"加载红外小目标数据集: {len(self.ids)} 张图像, {self.num_classes} 个类别")
        print(f"标注文件: {self.annotation_path}")

    def _build_transforms(self):
        """构建数据变换管道"""
        # 使用固定尺寸，因为RGBT-Tiny图像尺寸不一致
        image_size = (512, 640)  # 统一调整到这个尺寸
        
        if self.is_train and self.config.data.use_augmentation:
            # 训练时的数据增强
            transform_list = [
                T.RandomHorizontalFlip(self.config.data.horizontal_flip_prob),
                T.Resize(image_size),
            ]
            
            # 随机缩放裁剪 (针对小目标)
            if self.config.data.small_target_augmentation:
                transform_list.insert(1, T.RandomResizedCrop(
                    image_size, 
                    scale=(self.config.data.min_scale, self.config.data.max_scale)
                ))
            
            # 颜色抖动
            if self.config.data.color_jitter_prob > 0:
                transform_list.append(T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ))
            
            transform_list.extend([
                T.ToTensor(),
                T.Normalize(
                    mean=self.config.data.image_mean,
                    std=self.config.data.image_std
                )
            ])
            
            transforms = T.Compose(transform_list)
        else:
            # 验证/测试时的变换
            transforms = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=self.config.data.image_mean,
                    std=self.config.data.image_std
                )
            ])
        
        return transforms

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        
        # 加载图像
        img_info = coco.loadImgs(img_id)[0]
        img_file_name = img_info['file_name']
        
        # 使用RGBTinyConfig中的图像路径
        img_path = self.dataset_cfg.image_path / img_file_name
        img = Image.open(img_path).convert('RGB')
        
        # 加载标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # 解析边界框和标签
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # COCO格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # 转换为 [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        if len(boxes) == 0:
            # 如果没有目标，使用空数组
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
        
        # 原始图像尺寸
        orig_size = torch.tensor([img_info['height'], img_info['width']])
        
        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': orig_size
        }
        
        # 应用变换
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.ids)


def build_infrared_dataset(config: Config, is_train: bool = True):
    """构建红外小目标数据集"""
    
    dataset = InfraredSmallTargetDataset(
        config=config,
        is_train=is_train
    )
    
    return dataset


def collate_fn(batch):
    """自定义批次整理函数"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    return images, targets


# 测试函数
def test_dataset_loading():
    """测试数据集加载"""
    print("测试数据集加载...")
    
    try:
        from config import get_default_config
        config = get_default_config()
        
        # 检查数据集路径
        dataset_cfg = config.data.dataset
        dataset_cfg.check_paths()
        
        # 构建数据集
        train_dataset = build_infrared_dataset(config, is_train=True)
        val_dataset = build_infrared_dataset(config, is_train=False)
        
        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        # 测试一个样本
        if len(train_dataset) > 0:
            img, target = train_dataset[0]
            print(f"\n样本测试:")
            print(f"  图像尺寸: {img.shape}")
            print(f"  目标数量: {len(target['boxes'])}")
            print(f"  目标键: {list(target.keys())}")
            print("✓ 数据集加载测试通过!")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_dataset_loading()