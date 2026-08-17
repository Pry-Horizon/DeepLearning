"""
案例：
    演示张量的拼接操作
涉及到的API：
    cat()       不改变维度数，拼接张量，除了拼接的那个维度为，其他维度必须保持一致
    stack()     改变维度，拼接张量，张量的所有维度保持一致
"""

#导包
import torch

#1.创建两个张量
t1 = torch.randint(1,10,(2,3))
print(f"t1:{t1}")
t2 = torch.randint(1,10,(6,3))
print(f"t2:{t2}")

#2.演示张量的拼接
t3 = torch.cat([t1,t2],dim = 0)
print(f"t3:{t3},shape:{t3.shape}")
print("==" * 30)

t4 = torch.randint(1,10,(6,3))
t5 = torch.stack([t2,t4],dim = 0)
print(f"t5:{t5},shape:{t5.shape}")  #----->(2,6,3)

t6 = torch.stack([t2,t4],dim = 1)
print(f"t6:{t6},shape:{t6.shape}")