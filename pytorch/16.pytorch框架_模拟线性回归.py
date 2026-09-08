#导包
import torch
from torch.utils.data import TensorDataset  #构造数据集对象
from torch.utils.data import DataLoader     #数据加载器
from torch import nn                        #导入神经网络中的损失函数和假设函数
from torch import optim                     #optim中有优化器函数
from sklearn.datasets import make_regression#创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#1.定义函数，创建线性回归样本数据集

def create_dataset():
    #01.创建数据集对象
    x,y,coef = make_regression(n_samples = 100,
                               n_features = 1,
                               noise = 10,
                               coef = True,
                               bias = 14.5,
                               random_state = 3
                               )

    #02.把上述数据集对象封装成张量
    x = torch.tensor(x,dtype = torch.float)
    y = torch.tensor(y,dtype = torch.float)

    #03.返回结果
    return x,y,coef

#2.定义函数，表示模型训练

def train(x,y,coef):
    #01.创建数据集对象，把tenser->数据集对象->数据加载器
    dataset = TensorDataset(x,y)

    #02.创建数据加载器对象。
    dataloader = DataLoader(dataset,batch_size = 16,shuffle=True)  #参数：数据集对象 ，批次大小 ，是否打乱数据（训练集打乱，数据集不打乱）

    #03.创建初始的线性回归模型。
    model = nn.Linear(1,1) #参数：输入，输出

    #04.创建损失函数对象。
    criterion = nn.MSELoss()

    #05.创建优化器对象。
    optimizer = optim.SGD(model.parameters(),lr = 0.01)

    #06.具体的训练过程
        #6.1 定义变量，分别表示：训练轮数，每轮的损失，训练总损失值，训练的样本数
    epochs,loss_list,total_loss,total_sample = 100,[],0.0,0
        #6.2 开始训练，按轮训练
    for epoch in range(epochs):
        total_loss = 0.0
        total_sample = 0
        #6.3 每轮是分批次训练的，所以从数据加载器中获取批次数据
        for train_x,train_y in dataloader:  #(16,16,16,16,16,16,4)
            #6.4 模型预测
            y_pred = model(train_x)
            #6.5 计算每批平均损失
            loss = criterion(y_pred,train_y.reshape(-1,1))
            #6.6 计算总损失 和 样本（批次）数
            total_loss += loss.item()
            total_sample += 1
            #6.7 梯度清零 反向传播 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #6.8 把本轮的（平均）损失值，添加到列表中
        loss_list.append(total_loss/total_sample)
        print(f"轮数：{epoch + 1}，平均损失值：{total_loss/total_sample}")

    #07.打印100轮的平均训练结果
    print(f"100轮的平均损失分别为：{loss_list}")
    print(f"模型参数，权重：{model.weight},偏置：{model.bias}")

    #08.绘制损失曲线
    plt.plot(range(epochs),loss_list)
    plt.title("损失值曲线变化图")
    plt.grid()
    plt.show()

    #09.绘制预测值与真实值的关系
    #9.1 绘测样本点分布情况
    plt.scatter(x,y)
    #9.2 绘制训练模型的预测值
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])
    plt.plot(x,y_pred,color = 'green',label = '预测值')
    plt.plot(x,y_true,color = 'red',label = '真实值')
    plt.show()


if __name__ == '__main__':
    #创建数据集
    x,y,coef = create_dataset()
    print(f"x:{x},y:{y},coef:{coef}")
    #模型训练
    train(x,y,coef)


