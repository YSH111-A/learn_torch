import torch
import torchvision.models
from models_saved import  *

from P6_models.model_trained import vgg_16

#保存方式1
model1 = torch.load("vgg_16method1.pth")
print(model1)

#保存方式2
vgg_16 = torchvision.models.vgg16(weights = None)
#model = torch.load("vgg_16state_dict.pth")
vgg_16.load_state_dict(torch.load("vgg_16state_dict.pth"))
print(vgg_16)


