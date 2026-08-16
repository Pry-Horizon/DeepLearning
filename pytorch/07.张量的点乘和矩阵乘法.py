"""
案例:
    演示张量的点乘和矩阵乘法

点乘:
    要求:两个张量的维度一致,对应元素直接做相应操作
    API:
        t1*t2
        t1.mul(t2)

矩阵乘法:
    要求:两个张量,第一个张量的列数等于第二个张量的行数(A行=B列)
    结果:A行B列
    API:
        t1@t2
        t1.matmul(t2)
        t1.dot(t2)    #只针对一维张量

"""

#导包
import torch

#1.点乘
def dm01():
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[4,5,6],[1,2,3]])

    t3 = t2 * t1
    t4 = t2.mul(t1)

    print(f"t1:{t1}")
    print(f"t2:{t2}")
    print(f"t3 = t1 * t2 = {t3}")
    print(f"t4 = t1 * t2 = {t4}")
    

#2.矩阵乘法
def dm02():
    t1 = torch.Tensor(4,5)
    t2 = torch.Tensor(5,6)

    t3 = t1.matmul(t2)
    t4 = t1@t2

    print(f"t1:{t1}")
    print(f"t2:{t2}")
    print(f"t3 = t1 * t2 = {t3}")
    print(f"t4 = t1 * t2 = {t4}")

    


#.测试
if __name__ == '__main__':
    dm01()
    dm02()
