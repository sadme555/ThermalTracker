"""
红外小目标数据集加载器 - 修复 collate_fn 问题
"""

import os
import torch
import torch.utils.data
from pycocotools.coco import COCO
from PIL import Image
from typing import Dict, List, Optional, Tuple
import numpy as np
import torchvision.transforms as T

from .rgbt_tiny_config import RGBTinyConfig


class InfraredSmallTargetDataset(torch.utils.data.Dataset):
    def __init__(self, config, is_train: bool = True):
        """
        初始化数据集
        Args:
            config: 配置对象
            is_train: 是否为训练模式
        """
        self.config = config
        self.is_train = is_train
        
        # 使用现有的RGBTinyConfig
        self.dataset_cfg = RGBTinyConfig(config.data.dataset_root)
        
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
        image_size = getattr(self.config.data, 'image_size', (512, 640))
        
        # 获取数据增强配置，如果不存在则使用默认值
        use_augmentation = getattr(self.config.data, 'use_augmentation', True) and self.is_train
        horizontal_flip_prob = getattr(self.config.data, 'horizontal_flip_prob', 0.5)
        small_target_augmentation = getattr(self.config.data, 'small_target_augmentation', True)
        min_scale = getattr(self.config.data, 'min_scale', 0.8)
        max_scale = getattr(self.config.data, 'max_scale', 1.2)
        color_jitter_prob = getattr(self.config.data, 'color_jitter_prob', 0.5)
        image_mean = getattr(self.config.data, 'image_mean', [0.485, 0.456, 0.406])
        image_std = getattr(self.config.data, 'image_std', [0.229, 0.224, 0.225])
        
        if use_augmentation:
            # 训练时的数据增强
            transform_list = [
                T.RandomHorizontalFlip(horizontal_flip_prob),
                T.Resize(image_size),
            ]
            
            # 随机缩放裁剪 (针对小目标)
            if small_target_augmentation:
                transform_list.insert(1, T.RandomResizedCrop(
                    image_size, 
                    scale=(min_scale, max_scale)
                ))
            
            # 颜色抖动
            if color_jitter_prob > 0:
                transform_list.append(T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ))
            
            transform_list.extend([
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std)
            ])
            
            transforms = T.Compose(transform_list)
        else:
            # 验证/测试时的变换
            transforms = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std)
            ])
        
        return transforms

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        
        # 加载图像
        img_info = coco.loadImgs(img_id)[0]
        img_file_name = img_info['file_name']
        
        # 构建图像路径
        img_path = os.path.join(self.dataset_cfg.image_path, img_file_name)
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            # 尝试不同的路径格式
            img_path_alt = os.path.join(self.config.data.dataset_root, "images", img_file_name)
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path).convert('RGB')
        
        # 获取图像原始尺寸
        orig_width, orig_height = img.size
        
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
            # 转换为 [x1, y1, x2, y2] 并归一化到 [0, 1]
            x1 = x / orig_width
            y1 = y / orig_height
            x2 = (x + w) / orig_width
            y2 = (y + h) / orig_height
            
            # 确保坐标在 [0, 1] 范围内
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            boxes.append([x1, y1, x2, y2])
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
        orig_size = torch.tensor([orig_height, orig_width])
        
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


def build_infrared_dataset(config, is_train=True):
    """构建红外小目标数据集 - 修复版本"""
    try:
        dataset = InfraredSmallTargetDataset(config, is_train=is_train)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        # 尝试使用备用方法
        dataset = create_fallback_dataset(config, is_train)
    
    # 数据验证
    print("Validating dataset...")
    valid_count = 0
    for i in range(min(10, len(dataset))):  # 检查前10个样本
        try:
            image, target = dataset[i]
            
            # 检查图像
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(f"Sample {i}: Image contains NaN/Inf")
                continue
                
            # 检查边界框
            if 'boxes' in target:
                boxes = target['boxes']
                if len(boxes) > 0:
                    # 检查边界框坐标是否在[0,1]范围内
                    if (boxes < 0).any() or (boxes > 1).any():
                        print(f"Sample {i}: Box coordinates out of range [0,1]")
                        print(f"Box ranges: min={boxes.min()}, max={boxes.max()}")
                        continue
                    
                    # 检查边界框面积是否为正
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    if (areas <= 0).any():
                        print(f"Sample {i}: Invalid box areas")
                        continue
            
            valid_count += 1
            
        except Exception as e:
            print(f"Sample {i}: Error during validation - {e}")
            continue
    
    print(f"Data validation: {valid_count}/10 samples valid")
    return dataset


def create_fallback_dataset(config, is_train=True):
    """创建备用数据集 - 当主方法失败时使用"""
    print("Using fallback dataset creation method...")
    
    # 直接使用RGBTinyConfig创建数据集
    from .rgbt_tiny_config import RGBTinyConfig
    
    dataset_cfg = RGBTinyConfig(config.data.dataset_root)
    split = 'train' if is_train else 'test'
    annotation_path = dataset_cfg.get_annotation_path(split)
    
    # 创建简化的配置对象
    class FallbackConfig:
        class data:
            dataset_root = config.data.dataset_root
            image_size = getattr(config.data, 'image_size', (512, 640))
            use_augmentation = getattr(config.data, 'use_augmentation', True) and is_train
            horizontal_flip_prob = getattr(config.data, 'horizontal_flip_prob', 0.5)
            small_target_augmentation = getattr(config.data, 'small_target_augmentation', True)
            min_scale = getattr(config.data, 'min_scale', 0.8)
            max_scale = getattr(config.data, 'max_scale', 1.2)
            color_jitter_prob = getattr(config.data, 'color_jitter_prob', 0.5)
            image_mean = getattr(config.data, 'image_mean', [0.485, 0.456, 0.406])
            image_std = getattr(config.data, 'image_std', [0.229, 0.224, 0.225])
            num_classes = getattr(config.data, 'num_classes', 7)
        
        data = data()
    
    return InfraredSmallTargetDataset(FallbackConfig, is_train=is_train)


def collate_fn(batch):
    """
    自定义批次整理函数
    将图像堆叠成张量，目标保持为列表
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # 将图像列表堆叠成张量
    if len(images) > 0:
        images = torch.stack(images, dim=0)
    
    return images, targets

def get_transforms(config, is_train=True):
    """
    获取数据增强变换
    """
    if is_train:
        # 训练时的数据增强
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证/测试时的变换
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    return transform

# 测试函数
def test_dataset_loading():
    """测试数据集加载"""
    print("测试数据集加载...")
    
    try:
        # 创建简单的配置对象
        class SimpleConfig:
            class DataConfig:
                dataset_root = "datasets/RGBT-Tiny"
                image_size = (512, 640)
                use_augmentation = True
                horizontal_flip_prob = 0.5
                small_target_augmentation = True
                min_scale = 0.8
                max_scale = 1.2
                color_jitter_prob = 0.5
                image_mean = [0.485, 0.456, 0.406]
                image_std = [0.229, 0.224, 0.225]
            
            data = DataConfig()
        
        config = SimpleConfig()
        
        # 检查数据集路径
        dataset_cfg = RGBTinyConfig(config.data.dataset_root)
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
            
            # 测试 collate_fn
            batch = [(img, target), (img, target)]
            images, targets = collate_fn(batch)
            print(f"  批次图像尺寸: {images.shape}")
            print(f"  批次目标数量: {len(targets)}")
            print("✓ 数据集加载测试通过!")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False



if __name__ == '__main__':
    test_dataset_loading()