"""
案例：
    演示自动微分模块，具体如何求导

回顾：
    权重更新公式：
        w新 = w - 学习率*梯度
        梯度 = 损失函数对w求导
    
    pytorch有自动微分模块
"""

#导包
import torch

#1.定义变量记录初始权重w
w = torch.tensor(10,requires_grad=True,dtype = torch.float)

#2.定义loss变量表示损失函数
loss = 2*w**2     

#3.打印梯度函数类型
#print(f"梯度函数类型为：{loss.grad_fn}")

#4.计算梯度，梯度 = 损失函数的导数，计算完毕后，结果在w.grad里
loss.sum().backward()

#5.代入权重更新公式
w = w - 0.01*w.grad  

#6.打印最终结果
print(f"更新后的权重：{w}")


