"""
案例：
    演示神经网络搭建流程

深度学习的4个步骤：
    1. 数据准备
    2.搭建神经网络
    3.模型训练
    4.模型测试

神经网络搭建流程：
    1.定义一个神经网络类，继承nn.Module
    2.在_init__()方法中初始化网络层
    3.在forward()方法中定义前向传播过程

"""

#导包
import torch
import torch.nn as nn
from torchsummary import summary

#1.定义一个神经网络类，继承nn.Module
class ModeDemo(nn.Module):
    #1.1 在_init__()方法中初始化网络层
    def __init__(self):
        #(1).初始化父类成员。
        super().__init__()
        #(2).搭建神经网络
        #隐藏层1：
        self.linear1 = nn.Linear(3,3)
        #隐藏层2：
        self.linear2 = nn.Linear(3,2)
        #输出层：
        self.output = nn.Linear(2,2)
        #(3).对隐藏层进行参数初始化
        #隐藏层1：
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        #隐藏层2：
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #1.2 在forward()方法中定义前向传播过程
    def forward(self,x):
        #(1).第一层隐藏层计算：加权求和 + 激活函数
        x = self.linear1(x)  #加权求和
        x = torch.sigmoid(x) #激活函数
        #(2).第二层隐藏层计算：加权求和 + 激活函数
        x = torch.relu(self.linear2(x)) #加权求和 + 激活函数
        #(3).输出层计算：加权求和
        x = self.output(x)  #加权求和
        x = torch.softmax(x,dim=1) #激活函数
        return x


#2.模型训练

def train():
    #1.创建模型对象。
    my_model = ModeDemo()
    #2.创建数据集样本
    data = torch.randn(5,3)
    #3.调用神经网络模型 -> 进行模型训练
    output = my_model(data)  #底层自动调用了 forward()方法
    print(output)
    print('模型结构：')
    #4.打印模型参数
    for name,parameter in my_model.named_parameters():
        print(f"name:{name}\n")
        print(f"parameter:{parameter}\n")

#3.测试
if __name__ == '__main__':
    train()

    
