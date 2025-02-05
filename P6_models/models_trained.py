import torch
import torch.nn as nn
import torch.nn.functional as F
import  torchvision
from torchvision.models import vgg16, VGG16_Weights

train_set = torchvision.datasets.CIFAR10(
    root=r'../data_set',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

vgg16_false = torchvision.models.vgg16(weights=("pretrained",
                                                VGG16_Weights.IMAGENET1K_V1))
vgg16_true = torchvision.models.vgg16(weights=None)

# print(vgg16_false)
# print(vgg16_true)
##添加网络结构
vgg16_true.classifier.add_module(
    'add_linear',
    nn.Linear(1000,10)
)
print(vgg16_true)

##修改网络结构
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
