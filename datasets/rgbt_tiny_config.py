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
        """根据划分获取图片路径"""
        if split == 'train':
            split_file = self.train_split
        else:
            split_file = self.test_split
            
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
            
        image_paths = []
        for img_name in image_names:
            # 根据你的数据集结构调整路径
            # 假设图片路径格式为: DJI_0022_1/00/xxxx.jpg
            img_path = self.image_path / img_name
            if img_path.exists():
                image_paths.append(str(img_path))
                
        return image_paths

    def check_paths(self):
        """检查所有路径是否存在"""
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
        """获取标注文件路径"""
        if split == 'train':
            return str(self.annotation_path / "instances_00_train2017.json")
        else:
            return str(self.annotation_path / "instances_00_test2017.json")