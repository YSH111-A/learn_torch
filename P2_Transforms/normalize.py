from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")

img = Image.open(r"C:\Users\YSH\Pictures\0b17a32e56ee9db0eca7a533fc695f11161300378-vmake-vmake.png")

##Totensor
trans_Totensor  = transforms.ToTensor()
img_trans = trans_Totensor(img)
##normalize
#print(img_trans[0][0][0])
trans_norm = transforms.Normalize(mean=[0.4, 0.4, 0.4],std = [0.05,0.05, 0.05])
img_norm = trans_norm(img_trans)
#print(img_norm[0][0][0])
writer.add_image('trans_trans_norm', img_norm,2)

##ReSize
#print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
#img_resize ->totensor->img-resize tensor
img_resize = trans_Totensor(img_resize)
writer.add_image('trans_resize', img_resize,4)
#print(img_resize.size)

##Compose - resize -2
trans_resize_2  = transforms.Resize(512)#建立一个实例化resize,这个函数处理PIL数据，函数要输入到compose中
#实例化缩放函数compose要先输入两个实例化函数作为输入，resize和Totensor
#PIL -> PIL->tensor
trans_compose = transforms.Compose([trans_resize_2,trans_Totensor])
img_resize_2 = trans_compose(img)
writer.add_image('resize_2', img_resize_2,0)

#随机裁剪：RandomCrop
trans_random = transforms.RandomCrop((512,1024))
#实例化
trans_compose = transforms.Compose([trans_random,trans_Totensor])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image('random_crop', img_crop,i)
    print(img_crop.size)


writer.close()

