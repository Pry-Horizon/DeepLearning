"""
案例：
    演示张量的基本创建方式

张量：
    PyTorch框架 最常用的深度学习框架，ANN、CNN、RNN等都需要用到张量

    张量->存储同一类型元素的容器，且元素必须是数值

张量的基本创建方式:
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量，其也可以用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

细节：
    Tensor较于tensor可以根据形状创建
"""

#导包
import torch
import numpy as np

#1.定义函数，演示：torch.tensor 根据指定数据创建张量
def dm01():
    #场景1：标量 张量
    t1 = torch.tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print('-' * 30)

    #场景2：二维列表->张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f"t2:{t2},type:{type(t2)}")
    print('-' * 30)

    #场景3:numpy nd数组->张量
    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.tensor(data)
    print(f"t3:{t3},type:{type(t3)}")
    print('-' * 30)

    #场景4：尝试直接创建指定维度（例如：2行3列的张量）
    #t4 = torch.tensor(2,3)
    #print(f"t4:{t4},type:{type(t4)}")  会报错，只有大写可以支持形状

#2.定义函数，演示：torch.Tensor 根据形状创建张量，其也可以用来创建指定数据的张量
def dm02():
#场景1：标量 张量
    t1 = torch.Tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print('-' * 30)

    #场景2：二维列表->张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f"t2:{t2},type:{type(t2)}")
    print('-' * 30)

    #场景3:numpy nd数组->张量
    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.Tensor(data)
    print(f"t3:{t3},type:{type(t3)}")
    print('-' * 30)

    #场景4：尝试直接创建指定维度（例如：2行3列的张量）
    t4 = torch.IntTensor(2,3)
    print(f"t4:{t4},type:{type(t4)}")  #会报错，只有大写可以支持形状

#3.定义函数，演示：torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

#4.定义测试函数。
if __name__ == '__main__':
    dm01()
    dm02()