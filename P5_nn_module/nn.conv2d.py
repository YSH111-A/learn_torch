import  torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P1_Tensor_board.test_tensor_board import writer

dataset = torchvision.datasets.CIFAR10(
    r"D:\PyCharm\algorithm\learn_torch\data_set",
    train=False,transform=torchvision.transforms.ToTensor(),download=False)

dataloader = DataLoader(dataset,batch_size=64,shuffle=False)

class ShuangHe(nn.Module):
    def __init__(self):
        super(ShuangHe,self).__init__()
        self.conv1 = nn.Conv2d(3,6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x
shuanghe = ShuangHe()

writer = SummaryWriter(log_dir='../logs')
step =1
for data in dataloader:
    imgs,targets = data
    output =  shuanghe(imgs)
    print(imgs.shape)
    print(output.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images('input',imgs,step)
    #torch.Size([64, 6, 30, 30])
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images('output',output,step)
    step +=1
    ####fsa

