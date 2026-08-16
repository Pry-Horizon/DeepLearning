"""
案例：
    演示张量的形状操作
涉及到的API：
    reshape()       在不改变原始数据的情况下进行形状转换
    unsqueeze()     在指定轴上增加一个维度
    squeeze()       删除所有为一的维度
    transpose()     一次只能交换两个维度
    permute()       一次可以同时交换多个维度
    view()
    contiguous()
    is_contiguous()

"""

#导包
import torch

#1.演示reshape()函数,在不改变原始数据的情况下进行形状转换
def dm01():
    #1.定义一个2行3列的张量
    t1 = torch.randint(1,10,(2,3))
    print(f"t1:{t1},shape:{t1.shape},row:{t1.shape[0]},columns:{t1.shape[1],t1.shape[-1]}")#shape[-1]指最后一个值，这里是列
    #2.通过reshape()函数把t1-->3行2列
    t2 = t1.reshape(3,2)
    print(f"t2:{t2},shape:{t2.shape},row:{t2.shape[0]},columns:{t2.shape[1],t2.shape[-1]}")
    print("==" * 30)

    #3.尝试t1-->2行5列，无法完成，从6个元素-->10个元素，无法实现

#2.演示unsqueeze()函数和squeeze()函数
def dm02():
    #1.定义2行3列的张量
    t1 = torch.randint(1,10,(2,3))
    print(f"t1:{t1},shape:{t1.shape}")  #shape:(2,3)  2行3列
    #2.在0轴上添加一个维度。
    t2 = t1.unsqueeze(0)
    print(f"t2:{t2},shape:{t2.shape}")  #shape:(1,2,3) 1层2行3列
    #3.在1轴上添加一个维度
    t3 = t1.unsqueeze(1)
    print(f"t3:{t3},shape:{t3.shape}")  #shape:(2,1,3) 2层1行3列
    #4.在2轴上添加一个维度
    t4 = t1.unsqueeze(2)
    print(f"t4:{t4},shape:{t4.shape}")  #shape:(2,3,1) 2层3行1列
    #5.删除所有为1的维度
    t6 = torch.randint(1,10,(1,3,1,2,1,1))
    t7 = t6.squeeze()
    print(f"t7:{t7},shape:{t7.shape}")
    print("==" * 30)

#3.演示transpose()函数和permute()函数
def dm03():
    #1.定义张量
    t1 = torch.randint(1,10,(2,3,4))
    #2.改变维度(2,3,4)-->(3,2,4)
    t2 = t1.transpose(0,1)
    print(f"t1:{t1},shape:{t1.shape}")
    print(f"t2:{t2},shape:{t2.shape}")
    #3,改变维度(2,3,4)-->(4,2,3)
    t3 = t1.permute(2,0,1)
    print(f"t3:{t3},shape:{t3.shape}")

#4.演示viwe()函数、contiguous()函数、is_contiguous()函数



#5.测试
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    