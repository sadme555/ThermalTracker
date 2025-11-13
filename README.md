# ThermalTracker

1.在datasets下创建文件夹：
mkdir RGBT-Tiny
2.在RGBT-Tiny文件夹：
mkdir annotations_coco data_split images
3.从网上查找RGBT-Tiny数据集，并将相应数据拷贝到对应文件夹并unzip

环境配置部分：Linux、CUDA>=9.2、GCC>=5.4
4.conda create -n deformable_detr python=3.7 pip
5.conda init bash
6.conda activate deformable_detr
7.conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
8.pip install -r requirements.txt


