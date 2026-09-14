"""
案例：
    演示梯度下降优化方法
"""

import torch
import torch.nn as nn
from torch import optim

# 1.演示梯度下降优化方法->动量法（Momentum）


def dm01_momentum():
    # 1.初始化权重参数。
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数。
    criterion = ((w**2)/2.0)
    # 3.创建优化器。
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)
    # 4.计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    # 5.重复更新权重
    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    print("==="*30)


# 2.演示梯度下降优化方法：->自适应学习率（AdaGrad）

def dm02_adagrad():
    # 1.初始化权重参数。
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数。
    criterion = ((w**2)/2.0)
    # 3.创建优化器。
    optimizer = optim.Adagrad(params=[w], lr=0.01)
    # 4.计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    # 5.重复更新权重
    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    print("==="*30)


# 3.演示梯度下降优化方法：->RMSProp

def dm03_prop():
    # 1.初始化权重参数。
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数。
    criterion = ((w**2)/2.0)
    # 3.创建优化器。
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)
    # 4.计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    # 5.重复更新权重
    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    print("==="*30)

# 4.演示梯度下降优化方法：->Adam


def dm04_adam():
    # 1.初始化权重参数。
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数。
    criterion = ((w**2)/2.0)
    # 3.创建优化器。
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))
    # 4.计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    # 5.重复更新权重
    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')
    print("==="*30)


# 5.测试。
if __name__ == '__main__':
    dm01_momentum()
    dm02_adagrad()
    dm03_prop()
    dm04_adam()
