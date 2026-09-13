"""
案例：
    演示二分类任务的损失函数
"""
# 导包
import torch
import torch.nn as nn


def demo01():
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)
    y_pred = torch.tensor([0.6901, 0.5432, 0.2693], dtype=torch.float)
    criterion = nn.BCELoss()
    loss = criterion(y_pred, y_true)
    print(f'损失：{loss}')


if __name__ == '__main__':
    demo01()
