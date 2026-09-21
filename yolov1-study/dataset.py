import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# YOLOv1全局超参
S = 7   # 7×7网格
B = 2   # 每个网格预测2个框
C = 20  # VOC20个类别
class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


class VOCYoloDataset(Dataset):
    def __init__(self, root, year="2007", image_set="train", transform=None):
        """
        root：VOC数据集父目录
        image_set：train / val
        transform：图像预处理
        """
        self.root = root
        self.transform = transform
        # 读取图片id列表
        image_list_path = os.path.join(
            root, f"VOC{year}", "ImageSets", "Main", f"{image_set}.txt")
        with open(image_list_path, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        # 读取图片
        img_path = os.path.join(self.root, "VOC2007",
                                "JPEGImages", f"{img_id}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 读取xml标注
        xml_path = os.path.join(self.root, "VOC2007",
                                "Annotations", f"{img_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 初始化标签：7,7,30
        label = np.zeros((S, S, B*5 + C))

        # 遍历所有目标
        for obj in root.iter("object"):
            cls_name = obj.find("name").text
            cls_idx = class_names.index(cls_name)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # 归一化中心点、宽高（相对于原图）
            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            # 目标落在哪个网格
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)
            # 中心点相对于网格左上角的偏移
            cell_x = x_center * S - grid_x
            cell_y = y_center * S - grid_y

            # 一个网格只负责一个物体
            if label[grid_y, grid_x, 4] == 0:
                label[grid_y, grid_x, 0:5] = [
                    cell_x, cell_y, box_w, box_h, 1.0]
                label[grid_y, grid_x, 10 + cls_idx] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
