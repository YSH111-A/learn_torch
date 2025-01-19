import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False,
                                       download=False,transform=torchvision.transforms.ToTensor()
                                       )
dataloader = DataLoader(dataset,batch_size=4,shuffle=False)

#####CNN卷积网络
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
        #计算最有可能的类和其值
        # values,indics = torch.max(x,dim = 1)
        return x

loss = nn.CrossEntropyLoss()
shuanghe = Seq_()
for data in dataloader:

    imgs,labels = data
    output  = shuanghe(imgs)
    res_loss = loss(output, labels)
    #对计算的损失进行反向传播
    res_loss.backward()
    print(res_loss.item())
    ########


