import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(3,4)
        self.linear2 = nn.Linear(4,5)
        self.linear3 = nn.Linear(5,5)
        self.output = nn.Linear(5,2)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.kaiming_normal_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        x = self.output(x)
        x = torch.softmax(x,dim=1)
        return x

def train():
    my_model = SimpleModel()
    torch.manual_seed(1)
    data = torch.randn(10,3)
    output = my_model(data)
    print(f"output = {output},type = {type(output)}")

if __name__ == '__main__':
    train()