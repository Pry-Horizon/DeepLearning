import torch
import torch.nn as nn

S = 7
B = 2
C = 20


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_7 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4_8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_9 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4_10 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv5_2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv5_4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_5 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5_6 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=2, padding=1)

        self.conv6_1 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(1024 * S * S, 4096)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))

        self.leaky = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block1
        x = self.leaky(self.conv1(x))
        x = self.pool1(x)

        # Block2
        x = self.leaky(self.conv2(x))
        x = self.pool2(x)

        # Block3
        x = self.leaky(self.conv3_1(x))
        x = self.leaky(self.conv3_2(x))
        x = self.leaky(self.conv3_3(x))
        x = self.leaky(self.conv3_4(x))
        x = self.pool3(x)

        # Block4
        x = self.leaky(self.conv4_1(x))
        x = self.leaky(self.conv4_2(x))
        x = self.leaky(self.conv4_3(x))
        x = self.leaky(self.conv4_4(x))
        x = self.leaky(self.conv4_5(x))
        x = self.leaky(self.conv4_6(x))
        x = self.leaky(self.conv4_7(x))
        x = self.leaky(self.conv4_8(x))
        x = self.leaky(self.conv4_9(x))
        x = self.leaky(self.conv4_10(x))
        x = self.pool4(x)

        # Block5
        x = self.leaky(self.conv5_1(x))
        x = self.leaky(self.conv5_2(x))
        x = self.leaky(self.conv5_3(x))
        x = self.leaky(self.conv5_4(x))
        x = self.leaky(self.conv5_5(x))
        x = self.leaky(self.conv5_6(x))

        # Block6
        x = self.leaky(self.conv6_1(x))
        x = self.leaky(self.conv6_2(x))

        # Flatten + FC
        x = torch.flatten(x, 1)
        x = self.leaky(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        return x.reshape(-1, S, S, B*5 + C)


if __name__ == "__main__":
    model = YOLOv1()
    test_input = torch.randn(2, 3, 448, 448)
    out = model(test_input)
    print("输出shape：", out.shape)
