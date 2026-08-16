"""
案例：
    演示张量和numpy之间如何相互转换，以及如何从标量张量中提取内容

涉及到的API：
    场景1：张量->numpy nd数组对象
        张量对象.numpy()          共享内容
        张量对象.numpy().copy()   不共享内容,链式编程
    场景2：numpy nd数组-> 张量
        from_numpy()             共享内容
        torch.tensor(nd数组)     不共享内容
    场景3：从标量张量中提取内容
        标量张量.item()

掌握:
    张量->numpy:  张量对象.numpy()
    numpy->张量:  torch.tensor(nd数组)
    从标量张量中提取内容:  标量张量.item()
"""

import torch
import numpy as np

#1.张量->numpy
def dm01():
    t1 = torch.tensor([1,2,3,45,67,89])
    n1 = t1.numpy()        #共享内存
    n2 = t1.numpy().copy()   #不共享内存
    print(f"t1 = {t1},type = {type(t1)}")
    print(f"n1 = {n1},type = {type(n1)}")
    print(f"n2 = {n2},type = {type(n2)}")

    n1[0] = 100
    n2[0] = 10000
    print(f"t1 = {t1},type = {type(t1)}") #与n1共享内存
    print(f"n1 = {n1},type = {type(n1)}")
    print(f"n2 = {n2},type = {type(n2)}")   
    
#2.numpy->张量
def dm02():
    n1 = np.array([11,22,33])
    t1 = torch.tensor(n1)       #不共享内存
    t2 = torch.from_numpy(n1)   #共享内存
    #t2 = torch.from_numpy(n1).type(torch.float)  #类型转换会打断内存共享
    print(f"n1 = {n1},type = {type(n1)}")
    print(f"t2 = {t2},type = {type(t2)}")
    print(f"t1 = {t1},type = {type(t1)}")

    n1[0] = 1000000000
    print(f"n1 = {n1},type = {type(n1)}")
    print(f"t2 = {t2},type = {type(t2)}")
    print(f"t1 = {t1},type = {type(t1)}")

#3.从标量(只有一个值的张量)张量中提取内容
def dm03():
    t1 = torch.tensor(100)
    print(f"t1 = {t1},type = {type(t1)}")

    value = t1.item()
    print(f"value = {value},type = {type(value)}")


#4.测试函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
