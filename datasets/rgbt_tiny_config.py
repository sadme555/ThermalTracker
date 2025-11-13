'''
管理数据集RGBT-Tinyd的配置:
1.数据集的根目录
2.图像路径
3.标注路径
4.划分文件路径
'''

import os
from pathlib import Path

class RGBTinyConfig:
    def __init__(self, root_path="datasets/RGBT-Tiny"):
        self.root_path = Path(root_path)
        self.image_path = self.root_path / "images"
        self.annotation_path = self.root_path / "annotations_coco"
        self.split_path = self.root_path / "data_split"
        
        # 训练和测试划分文件
        self.train_split = self.split_path / "train.txt"
        self.test_split = self.split_path / "test.txt"
        
    def get_image_paths(self, split='train'):
        if split == 'train':
            split_file = self.train_split
        else:
            split_file = self.test_split
            
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
            
        image_paths = []
        for img_name in image_names:
            img_path = self.image_path / img_name
            if img_path.exists():
                image_paths.append(str(img_path))
                
        return image_paths

    def check_paths(self):
        paths = {
            'root': self.root_path,
            'images': self.image_path,
            'annotations': self.annotation_path,
            'splits': self.split_path
        }
        
        for name, path in paths.items():
            exists = path.exists()
            print(f"{name}路径: {path} - {'存在' if exists else '不存在'}")
            if exists:
                items = list(path.glob('*'))
                print(f"  内容: {[item.name for item in items[:3]]}")

    def get_annotation_path(self, split='train'):
        if split == 'train':
            return str(self.annotation_path / "instances_00_train2017.json")
        else:
            return str(self.annotation_path / "instances_00_test2017.json")