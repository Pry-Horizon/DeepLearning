"""
案例：
    演示学习率衰减策略
    较之于AdaGrad、RMSProp、Adam可以手动控制学习率的调整

"""
import torch
from torch import optim
import matplotlib.pyplot as plt

# 1.演示等间隔学习率衰减


def dm01():
    # 1.记录初始的 学习率、训练轮数、每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10
    # 2.创建数据集：y_true、x、w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 3.创建优化器对象
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 4.创建学习率衰减对象
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    lr_list, epoch_list = [], []
    # 循环训练
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(iteration):
            y_pred = w*x
            loss = (y_pred - y_true)**2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        scheduler.step()
    print(f'lr_list:{lr_list},\nepoch_list:{epoch_list}')

    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()


# 2.演示指定间隔学习率衰减

def dm02():
    # 1.记录初始的 学习率、训练轮数、每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10
    # 2.创建数据集：y_true、x、w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 3.创建优化器对象
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 4.创建学习率衰减对象
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[44, 88, 199], gamma=0.5)
    lr_list, epoch_list = [], []
    # 循环训练
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(iteration):
            y_pred = w*x
            loss = (y_pred - y_true)**2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        scheduler.step()
    print(f'lr_list:{lr_list},\nepoch_list:{epoch_list}')

    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()


# 3.演示指数学习率衰减

def dm03():
    # 1.记录初始的 学习率、训练轮数、每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10
    # 2.创建数据集：y_true、x、w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float)
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 3.创建优化器对象
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 4.创建学习率衰减对象
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    lr_list, epoch_list = [], []
    # 循环训练
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        lr_list.append(scheduler.get_last_lr())

        for batch in range(iteration):
            y_pred = w*x
            loss = (y_pred - y_true)**2
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        scheduler.step()
    print(f'lr_list:{lr_list},\nepoch_list:{epoch_list}')

    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()


# 4.测试。
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
