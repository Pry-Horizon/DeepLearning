import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset():
    x,y,coef = make_regression(n_samples = 100,
                               n_features = 1,
                               noise = 10,
                               bias = 2.5,
                               coef = True,
                               random_state = 1)

    x = torch.tensor(x,dtype = torch.float)
    y = torch.tensor(y,dtype = torch.float)
    y = y.reshape(-1,1)

    return x,y,coef

def train(x,y,coef):
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset,batch_size = 16,shuffle = True)
    model = nn.Linear(1,1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr = 0.01)


if __name__ == '__main__':
    create_dataset()

