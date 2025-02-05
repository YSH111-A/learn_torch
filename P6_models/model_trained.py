import torch
import  torchvision
from torchvision.models import VGG16_Weights

vgg_16 = torchvision.models.vgg16(weights=("pretrained",
                                           VGG16_Weights.IMAGENET1K_V1))
#print(vgg_16)
##保存方式1:模型结构+模型参数
torch.save(vgg_16,"vgg_16method1.pth")

##保存方式2：模型参数
torch.save(vgg_16.state_dict(),"vgg_16state_dict.pth")
