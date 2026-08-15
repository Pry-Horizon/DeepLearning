"""
实例：
    创建指定类型的张量
涉及函数：
    type(torch支持的数据类型)
    half()/double()/float()/short()/int()/long()

"""

#导包
import torch

#场景1.创建指定类型的张量
t1 = torch.tensor([1,2,3,4,5,6,7,8],dtype = torch.float) #默认float 32
print(f"t1 = {t1}\n  dtype = {t1.dtype}\n  type = {type(t1)}\n")  #t1是什么   t1元素的类型    t1的张量类型

#场景2.创建好张量后->做类型转换
t2 = t1.type(torch.int16)
print(t2.dtype)
t3 = t1.type(torch.double)
print(t3.dtype)

#思路2.half()/double()/float()/short()/int()/long()
print(t2.half())
print(t2.double())
print(t2.int())
print(t2.long())
print(t2.short())