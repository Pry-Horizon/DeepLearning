"""
案例：
    演示自动微分的真实应用场景

结论：
    1.前向传播,计算出预测值z
    2.基于损失函数,结合预测值z 和真实值y,来计算梯度
    3.结合权重更新公式 w.data = w.data - a*w.grad  来更新权重

"""
import torch

#1.定义x，表示：特征(输入数据)，假设2行5列，全1矩阵
x = torch.ones(2,5,dtype = float)
print(f"初始输入x为:{x}")

#2.定义y，表示：真实值，假设：2行3列，全0矩阵
y = torch.zeros(2,3,dtype = float)
print(f"y:{y}")

#3.初始化可自动微分的权重 和 偏置
w = torch.randn(5,3,requires_grad=True,dtype = float)
b = torch.randn(2,3,requires_grad=True,dtype = float)

criterion = torch.nn.MSELoss()  #nn是neural network:神经网络，里边定义了许多损失函数

#4.前向传播计算出预测值z
for i in range(1,501):
    z = torch.matmul(x,w) + b

#5.定义损失函数。
    loss = criterion(z,y)           #loss 是计算出的损失

#6.进行自动微分，求导，结合反向传播，更新权重

    loss.sum().backward()
    w.data = w.data - 0.01*w.grad
    b.data = b.data - 0.01*b.grad
    w.grad.zero_()
    b.grad.zero_()
    print(f"第{i}次的参数：损失loss:{loss}  w梯度：{w.grad} b梯度：{b.grad}")
print(f"最终的参数：w:{w}\nb:{b}\n损失loss:{loss}\nw梯度：{w.grad}\nb梯度：{b.grad}")