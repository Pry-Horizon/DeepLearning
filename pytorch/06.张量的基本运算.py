"""
案例:
    演示张量的基本运算.

涉及到的API:
    add(),sub(),mul(),div(),neg()  -> 加减乘除,取反,
    add_(),sub_(),mul_(),div_(),neg_()  ->  功能同时,可以修改源数据
"""

#导包
import torch

t1 = torch.tensor([1,2,3])
t2 = t1.add(10)  #与 t2 = t1 + 10 等价

print(f"t1:{t1}")
print(f"t2:{t2}")

t3 = t1.add_(1000)
print(f"t1:{t1}")
print(f"t3:{t3}")

#其他函数效果同上

