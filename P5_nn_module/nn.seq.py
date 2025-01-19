import torch
from torch import  nn
from torch.nn import Conv2d, Flatten, MaxPool2d, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Seq_(nn.Module):
    def __init__(self):
        super(Seq_, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model1(x)
        values,indics = torch.max(x,dim = 1)
        return values,indics
shuanghe = Seq_()
print(shuanghe)
input = torch.randn(64, 3, 32, 32)
output = shuanghe(input)
print(output)

writer  = SummaryWriter("../logs_seq")
writer.add_graph(shuanghe, input)
writer.close()
