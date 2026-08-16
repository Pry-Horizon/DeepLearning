"""
案例:
    演示张量的常用运算函数

涉及到的API:
    sum(),max(),min(),mean()                  -->都有dim参数,0表示列,1表示行
    pow(),sqrt(),exp(),log(),log2(),log10()   -->没有dim参数

掌握的函数:
    sum(),max(),min(),mean(),pow()

"""

#导包
import torch

#1.记录初值
t1 = torch.tensor([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

print(f"t1:{t1}")

#2.有dim参数的函数
print(t1.sum(dim = 1))#行求和
print(t1.sum(dim = 0))#列求和
print(t1.sum())#全局求和
print("==" * 30)

print(t1.max(dim = 1))#行最大
print(t1.max(dim = 0))#列最大
print(t1.max())#全局最大
print("==" * 30)

#mean()计算平均值
t1 = t1.type(torch.float)
print(t1.mean(dim = 1))#行平均
print(t1.mean(dim = 0))#列平均
print(t1.mean())
print("==" * 30)

#3.没有dim参数的函数
print(t1.pow(2))#平方
print(t1**3)#立方
print("==" * 30)

print(t1.sqrt())#平方根
print("==" * 30)

print(t1.exp())#e的t1次幂
print("==" * 30)

print(t1.log())
print(t1.log2())
print(t1.log10())