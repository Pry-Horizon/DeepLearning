"""
案例：
    PyTorch中演示如何创建 线性 和 随机张量

涉及到函数：
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed()和torch.random.manual_seed() 随机种子设置
    torch.rand/randn() 创建随机浮点类型张量
    torch.randint(low,high,size=()) 创建随机整数类型张量

要掌握的函数：
    arange()  linspace()  manual_seed()  randint()

"""
#导包
import torch

#1.定义函数,演示: 创建线性张量
def dm01():
    #场景1：创建指定范围的线性张量
    t1 = torch.arange(0,10,2)  #0,2,4,6,8
    print(f"t1:{t1},type:{type(t1)}")
    print("==" * 30)
    #场景2：指定范围的线性等差数列张量
    t2 = torch.linspace(0,100,51)
    print(f"t2={t2},type={type(t2)}")
    print("==" * 30)

#2.定义函数,演示:创建随机张量
def dm02():
    #step1：设置随机种子
    torch.initial_seed()  #默认采用系统时间戳
    torch.manual_seed(4)  #每次随机数一致

    #step2：
    #场景1：均匀分布的（0，1）随机变量
    t1 = torch.rand(2,3)
    print(f"t1 = {t1},type = {type(t1)}")
    print("==" * 30)
    #场景2：符合正态分布的随机张量
    t2 = torch.randn(2,3)
    print(f"t2 = {t2},type = {type(t2)}")
    print("==" * 30)    
    #场景3：创建随机整数张量
    t3 = torch.randint(1,10,size=(3,5))
    print(f"t3 = {t3},type = {type(t3)}")
    print("==" * 30)

#3.测试函数
if __name__ == '__main__':
    dm01()
    dm02()
