"""
案例：
    演示如何创建全0全1和指定值的张量

涉及函数：
    torch.ones,torch.ones_like      全1张量
    torch.zeros,torch.zeroa_like    全0张量
    torch.full,torch.full_like      定值张量

需要掌握：
    zeros(),full()

"""

#导包
import torch

#场景1：全1张量
t1 = torch.ones(2,3)
print(f"t1 = {t1},type = {type(t1)}")
print("==" * 30)

    #t3是基于t2的形状，创建的全1张量
t2 = torch.tensor([[1,2],[3,3],[5,6]])
t3 = torch.ones_like(t2) 
print(f"t3 = {t3},type = {type(t3)}")
print("==" * 30)

#场景2：全0张量
t1 = torch.zeros(2,3)
print(f"t1 = {t1},type = {type(t1)}")
print("==" * 30)

    #t3是基于t2的形状，创建的全0张量
t2 = torch.tensor([[1,2],[3,3],[5,6]])
t3 = torch.zeros_like(t2) 
print(f"t3 = {t3},type = {type(t3)}")
print("==" * 30)

#场景3：定值张量
t1 = torch.full((2,3),255)
print(f"t1 = {t1},type = {type(t1)}")
print("==" * 30)

    #t3是基于t2的形状，创建的定值张量
t2 = torch.tensor([[1,2],[3,3],[5,6]])
t3 = torch.full_like(t2,255) 
print(f"t3 = {t3},type = {type(t3)}")
print("==" * 30)