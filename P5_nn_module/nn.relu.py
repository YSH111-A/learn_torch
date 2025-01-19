import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU,Sigmoid

# input = torch.tensor([
#     [1,-0.5],
#     [-1,3]
# ])

# input = torch.reshape(input, (-1,1,2,2))
# print(input)
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data_set", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

class relu1(torch.nn.Module):
    def __init__(self):
        super(relu1, self).__init__()
        self.relu_1 = torch.nn.ReLU()
        self.sigmoid_1 = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid_1(input)
        return output

shuanghe = relu1()

writer = SummaryWriter("../logs_relu")
step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("sigmod_input", imgs, global_step=step)
    output  = shuanghe(imgs)
    writer.add_images("sigmod_output", output, global_step=step)
    step += 1


writer.close()

print("f")
