"""
案例：
    演示多分类任务交叉熵损失函数。

损失函数介绍：
    1、概述：
        损失函数也叫成本函数、目标函数、代价函数、误差函数是机器学习中用于衡量模型预测值与真实值之间差异的函数。
    它在训练过程中起到指导模型优化的作用。
    2、分类问题：
        多分类交叉熵损失函数（CrossEntropyLoss）是用于多分类问题的常用损失函数。
        二分类交叉熵损失函数（BinaryCrossEntropyLoss）是用于二分类问题的常用损失函数。
    3、回归问题：
        MAE：平方误差损失函数（Mean Absolute Error Loss）是用于回归问题的常用损失函数。
        MSE：均方误差损失函数（Mean Squared Error Loss）是用于回归问题的常用损失函数。
        Smooth L1 Loss：平滑L1损失函数（Smooth L1 Loss）是用于回归问题的常用损失函数。

多分类任务交叉熵损失：CrossEntropyLoss
    涉及思路：
        Loss = -sum(y_true * log(y_pred))
        y_pred = S(f(x))
    记忆：
        x: 输入数据
        f(x): 加权和偏置后的输出
        S(f(x)): 激活函数
        y_true: 真实标签
        y_pred: 预测概率

"""
#导包
import torch
import torch.nn as nn

#1、定义函数，演示：多分类交叉熵损失。
def demo01():
    #1、手动创建样本的真实值
    y_true = torch.tensor([[0,1,0],[1,0,0]],dtype = torch.float)
    #2、手动创建样本的预测值
    y_pred = torch.tensor([[0.1,0.8,0.1],[0.7,0.2,0.1]],requires_grad=True,dtype=torch.float)
    #3、创建多分类交叉熵损失函数。
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred,y_true)
    print(f'损失为：{loss}')




#2、测试。
if __name__ == '__main__':
    demo01()
