"""
案例：
    演示自动微分模块循环实现，计算梯度，更新参数

需求：
    求y = x**2 + 20 的极小值点 并打印y是最小值时w的值(梯度)

解题步骤：
    1.定义 x = 10 requires_grad = True dtype = float
    2.定义函数 y = x**2 + 20
    3.利用梯度下降算法 循环迭代100次,求最优解
        3.1 正向传播
        3.2 梯度清零 x.grad.zero_()
        3.3 梯度更新 x.data = x.data - 0.01*x.grad

"""

import torch

w = torch.tensor(10,requires_grad = True,dtype = float)
print(f"权重初始值为:w ={w}")
loss = w**2 + 20

#3.循环实现100次
for i in range(1,570):

    #3.1 正向传播
    loss = w**2 + 20

    #3.2 反向传播,梯度更新 x.data = x.data - 0.01*x.grad
    loss.sum().backward()
    w.data = w.data - 0.01 * w.grad

    #3.3 梯度清零 x.grad.zero_()
    w.grad.zero_()

    #3.4 打印
    print(f"第{i}次权重值：{w:.5f}   0.01*w.grad : {0.01*w.grad:.5f}")

#4.打印最终结果
print(f"权重:{w:.5f}\n梯度:{w.grad:.5f}\nloss:{loss:.5f}")