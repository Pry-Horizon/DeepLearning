"""
案例：
    回归任务损失函数
"""
import torch
import torch.nn as nn


def demo01():
    y_true = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float)
    y_pred = torch.tensor(
        [1.0, 2.5, 3.9], dtype=torch.float, requires_grad=True)
    criterion = nn.L1Loss()
    loss = criterion(y_pred, y_true)
    print(f'MAE损失:{loss}')


def demo02():
    y_true = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float)
    y_pred = torch.tensor(
        [1.0, 2.5, 3.9], dtype=torch.float, requires_grad=True)
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    print(f'MSE损失:{loss}')


def demo03():
    y_true = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float)
    y_pred = torch.tensor(
        [1.0, 2.5, 3.9], dtype=torch.float, requires_grad=True)
    criterion = nn.SmoothL1Loss()
    loss = criterion(y_pred, y_true)
    print(f'平滑L1损失:{loss}')


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
