from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('../logs')
image_path = "../dataset/train/ants/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("ants", img_array,2,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y= x^2",i**2,i)

writer.close()

