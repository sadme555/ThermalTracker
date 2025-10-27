import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

# 使用绝对导入
from datasets.rgbt_tiny_config import RGBTinyConfig

class InfraredSmallTargetDataset(Dataset):
    def __init__(self, root="datasets/RGBT-Tiny", split='train', transforms=None):
        self.config = RGBTinyConfig(root)
        self.split = split
        self.transforms = transforms
        
        # 检查路径
        print("检查数据集路径...")
        self.config.check_paths()
        
        # 加载COCO标注
        ann_file = self.config.get_annotation_path(split)
        print(f"加载标注文件: {ann_file}")
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"标注文件不存在: {ann_file}")
            
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        print(f"数据集加载成功: {len(self.ids)} 张图像")
        
        # 预处理类别信息
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = [cat['id'] for cat in self.categories]
        self.category_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        
        print(f"类别信息: {[cat['name'] for cat in self.categories]}")
        
    def __len__(self):
        return len(self.ids)
    
    def _find_image_path(self, file_name):
        """根据文件名查找图像路径"""
        # 尝试直接路径
        img_path = self.config.image_path / file_name
        if img_path.exists():
            return img_path
            
        # 尝试只使用基本文件名
        base_name = os.path.basename(file_name)
        img_path = self.config.image_path / base_name
        if img_path.exists():
            return img_path
            
        # 尝试递归搜索
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            pattern = f"*{os.path.splitext(base_name)[0]}*{ext}"
            found = list(self.config.image_path.rglob(pattern))
            if found:
                return found[0]
                
        # 如果还是找不到，尝试在子目录中搜索
        for subdir in self.config.image_path.iterdir():
            if subdir.is_dir():
                potential_path = subdir / base_name
                if potential_path.exists():
                    return potential_path
                    
        return None
        
    def __getitem__(self, index):
        # 获取图像ID
        img_id = self.ids[index]
        
        # 加载图像信息
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # 查找图像路径
        img_path = self._find_image_path(file_name)
        
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"无法找到图像: {file_name}")
            
        print(f"加载图像 {index}: {img_path}")
        
        # 加载图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # 解析标注
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # COCO标注格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # 转换为 [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_to_idx[ann['category_id']])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'image_id': torch.tensor([img_id]),
            'boxes': boxes,
            'labels': labels,
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64),
            'orig_size': torch.as_tensor([image.shape[0], image.shape[1]])
        }
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 应用数据增强
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

def build_dataset(split='train', transforms=None):
    """构建数据集"""
    return InfraredSmallTargetDataset(split=split, transforms=transforms)