"""
案例：
    演示参数初始化的7种方法

参数初始化的目的：
    1. 避免梯度消失和梯度爆炸
    2. 加速模型收敛
    3.打破对称性

参数初始化的方式
    无法打破对称性的：
        全0初始化、全1初始化、全随机初始化
    可以打破对称性的：
        均匀分布初始化、正态分布初始化、Xavier初始化(非relu激活函数)、Kaiming初始化(relu激活函数)
"""

#导包
import torch
import torch.nn as nn

#1.均匀分布随机初始化
def test01():
    #创建一个线性层
    linear = nn.Linear(5,3)
    #使用均匀分布初始化权重和偏置
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print("="*30)

#2.固定值初始化
def test02():
    linear = nn.Linear(5,3)
    nn.init.constant_(linear.weight,5)
    nn.init.constant_(linear.bias,2)
    print(linear.weight.data)
    print("="*30)

#3.全0初始化
def test03():
    linear = nn.Linear(5,3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print("="*30)

#4.全1初始化
def test04():
    linear = nn.Linear(5,3)
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)
    print("="*30)

#5.正态分布初始化
def test05():
    linear = nn.Linear(5,3)
    nn.init.normal_(linear.weight,mean=0,std=1)
    nn.init.normal_(linear.bias,mean=0,std=1)
    print(linear.weight.data)
    print("="*30)

#6.kaiming初始化
def test06():
    linear = nn.Linear(5,3)
    nn.init.kaiming_normal_(linear.weight,mode='fan_in',nonlinearity='relu')
    print(linear.weight.data)
    print("="*30)

#7.xavier初始化
def test07():   
    linear = nn.Linear(5,3)
    nn.init.xavier_normal_(linear.weight,gain=1)
    print(linear.weight.data)
    print("="*30)

#8.测试
if __name__ == '__main__':
    test01()
    test02()
    test03()
    test04()
    test05()
    test06()
    test07()
