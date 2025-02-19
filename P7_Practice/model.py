import torch
from torch import nn

##搭建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5,stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self, x):
        x = self.model(x)
        return x
if __name__ == '__main__':
    net = Net()
    input = torch.randn(64,3,32,32)
    output = net(input)
    print(output.size())
    print(output.shape)