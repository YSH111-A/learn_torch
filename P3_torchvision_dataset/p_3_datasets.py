import torchvision
from torch.utils.tensorboard import SummaryWriter

from P1_Tensor_board.test_tensor_board import writer

data_set_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
##将dataet_transfrom应用到每一张图片，compose作用是进行一系列变换操作
train_set = torchvision.datasets.CIFAR10\
    (root='./data_set', train=True, transform=data_set_transform, download=True)
test_set = torchvision.datasets.CIFAR10 \
    (root='./data_set', train=False, transform=data_set_transform,download=True)


# print(train_set[0])
# print(test_set.classes)
#
# img,target = train_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("")
for i in range(10):
    img,target = test_set[i]
    writer.add_image('tast_set',img,i)

writer.close()
