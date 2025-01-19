import torch
import torch.nn as nn

input = torch.rand(5,5)*10
input = torch.reshape(input,(-1,1,5,5))
print(input)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,x):
        output = self.maxpool1(x)
        return output

net = Net()
output1 = net(input)
print(output1)