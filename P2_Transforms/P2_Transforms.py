from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


##python的用法==>>tensor数据张量
##通过transforms.Totensor去看两个问题
#1、transforms该如何使用（python）
#2、Tensor数据类型

img_path = r"/dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

tensor_trans = transforms.ToTensor()
##Totensor将图片的格式PIL Image或numpy.ndarray转换成张量
tensor_img = tensor_trans(img)

writer.add_image("数据名称",tensor_img)
writer.close()

