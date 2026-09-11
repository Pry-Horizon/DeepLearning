import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from torch import optim

def create_dataset():
    x,y,coef = make_regression(n_samples = 100,
                               n_features = 1,
                               noise = 10,
                               bias = 14.5,
                               coef = True,
                               random_state = 3)

    x = torch.tensor(x,dtype = torch.float)
    y = torch.tensor(y,dtype = torch.float)
    y = y.reshape(-1,1)

    return x,y,coef

def train(x,y,coef):
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset,batch_size = 10,shuffle = True)
    model = nn.Linear(1,1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr = 0.01)

    epochs = 1000 
    i = 0
    loss_list = []
    for i in range(epochs):
        total_loss = 0.0
        total_sample = 0
        for train_x,train_y in dataloader:
            pred_y = model(train_x)
            loss = criterion(train_y,pred_y)
            total_loss += loss
            total_sample += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(total_loss/total_sample)
        print(f'轮数：{i},本轮的平均损失为：{total_loss/total_sample}')


if __name__ == '__main__':
    x,y,coef = create_dataset()
    train(x,y,coef)




    