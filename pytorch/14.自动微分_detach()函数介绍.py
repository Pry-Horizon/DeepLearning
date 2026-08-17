"""
案例：
    演示detach()函数的功能，解决自动微分的弊端

回顾：
    自动微分 = 求导，即基于损失函数计算梯度。
    w.data = w.data - a*w.grad

问题：
    一个张量一旦设置了自动微分，这个张量就不能直接转成numpy的nd数组了，需要通过detach()函数解决
"""

#导包
import torch
import numpy as np

#1.定义张量,设置了自动微分
t1 = torch.tensor([10,20],requires_grad = True,dtype = torch.float)
print(f"t1:{t1},type:{type(t1)}")

#2.转numpy报错
"""
n1 = t1.numpy()
print(f"t2:{t2},type:{type(t2)}")
"""

#3.解决方法，通过detach()函数，拷贝一份张量，然后再进行转换
t2 = t1.detach()
print(f"t2:{t2},type:{type(t2)}") #t1和t2共享内存
n2 = t2.numpy()
print(f"n2:{n2},type:{type(n2)}")
n3 = t1.detach().numpy()
print(f"n3:{n3},type:{type(n3)}")
print("==" * 30)

#4.detach()共享内存
t1.data[0] = 10000
print(t1)
print(t2)
print(n2)
print(n3)